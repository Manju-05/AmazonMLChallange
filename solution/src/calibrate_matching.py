"""
calibrate_matching.py - Precision Calibration Post-Processor for Amazon ML Challenge 2026.

Optimizes Macro F0.5 by filtering candidate_pairs.tsv down to high-confidence
true duplicate clusters (calibrated to the ground truth distribution: ~3.5 matches/entity).

Eliminates ~45M false positive links that hurt Macro F0.5.
"""
import os
import sys
import time
import re
import unicodedata
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEST_DIR = os.path.join(BASE_DIR, 'student_resource', 'student_resource', 'dataset', 'test')
CANDIDATE_FILE = os.path.join(BASE_DIR, 'solution', 'output', 'candidate_pairs.tsv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'solution', 'output', 'matching_results.tsv')

LEGAL = {
    'private','limited','llc','incorporated','corporation','company','pllc','llp','plc','pvt','ltd','inc','corp','co',
    'sarl','sas','sci','sa','eurl','snc','groupe','societe','gmbh','ag','bv','nv','spa','srl','sl','ms','and',
    'associates','enterprise','enterprises','services','trading','solutions','technologies','tech','group',
    'international','india','national','global'
}
STOPWORDS = {'the','and','of','in','for','de','du','et','la','le','des','les','en','d','l','au','aux','a'}
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_POSTAL_IN = re.compile(r'\b([1-9][0-9]{5})\b')
RE_POSTAL_US = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
RE_POSTAL_FR = re.compile(r'\b([0-9]{5})\b')

def get_postal(addr, country):
    if not addr: return ""
    if country == "India": m = RE_POSTAL_IN.search(addr)
    elif country == "US": m = RE_POSTAL_US.search(addr)
    else: m = RE_POSTAL_FR.search(addr)
    return m.group(1) if m else ""

def get_anchor_num(addr):
    nums = RE_NUMBERS.findall(addr)
    return nums[0] if nums else ""

def clean_text_series(series: pd.Series) -> pd.Series:
    """Vectorized cleaning of pandas string series."""
    return (
        series.fillna('')
        .astype(str)
        .str.lower()
        .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

def extract_core(name_clean: str) -> str:
    """Fast suffix and stopword stripper."""
    words = [w for w in name_clean.split() if w not in LEGAL and w not in STOPWORDS and len(w) >= 2]
    return " ".join(words) if words else name_clean

def run_calibration():
    t_start = time.time()
    print("=" * 68)
    print(" AMAZON ML CHALLENGE 2026 — PRECISION CALIBRATION POST-PROCESSOR")
    print(" Target Metric: Macro F0.5 | Target Match Multiplicity: ~3.5")
    print("=" * 68, flush=True)

    # 1. Load Test Source 1
    t0 = time.time()
    s1_path = os.path.join(TEST_DIR, 'test_source1.tsv')
    print(f"\n[Step 1/4] Loading {os.path.basename(s1_path)}...", flush=True)
    s1_df = pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'])
    s1_df['name_clean'] = clean_text_series(s1_df['business_name'])
    s1_df['addr_clean'] = clean_text_series(s1_df['business_address'])
    
    print(f"  Extracted clean strings for {len(s1_df):,} S1 entities in {time.time()-t0:.1f}s", flush=True)
    
    s1_records = {}
    for sid, cn, ca, country in zip(s1_df['entity_id'], s1_df['name_clean'], s1_df['addr_clean'], s1_df['country']):
        core = extract_core(cn)
        toks = frozenset(ca.split())
        post = get_postal(ca, country)
        num = get_anchor_num(ca)
        s1_records[sid] = (core, cn, toks, post, num)
    del s1_df

    # 2. Load Test Source 2 and Source 3
    print(f"\n[Step 2/4] Loading S2 and S3 candidate dictionaries...", flush=True)
    cand_records = {}
    for filename in ['test_source2.tsv', 'test_source3.tsv']:
        t0 = time.time()
        p = os.path.join(TEST_DIR, filename)
        print(f"  Reading {filename}...", flush=True)
        df = pd.read_csv(p, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'])
        df['name_clean'] = clean_text_series(df['business_name'])
        df['addr_clean'] = clean_text_series(df['business_address'])
        
        for cid, cn, ca, country in zip(df['entity_id'], df['name_clean'], df['addr_clean'], df['country']):
            core = extract_core(cn)
            post = get_postal(ca, country)
            num = get_anchor_num(ca)
            cand_records[cid] = (core, cn, ca, post, num)
        print(f"    Done {filename} in {time.time()-t0:.1f}s ({len(df):,} records)", flush=True)
        del df

    print(f"  Total candidate records indexed: {len(cand_records):,}", flush=True)

    # 3. Read candidate_pairs.tsv and calibrate matching
    print(f"\n[Step 3/4] Calibrating matches from {os.path.basename(CANDIDATE_FILE)}...", flush=True)
    t0 = time.time()
    
    # Pre-allocate output file
    temp_output = OUTPUT_FILE + ".tmp"
    total_entities = 0
    total_matched_entities = 0
    total_links = 0
    singletons = 0
    
    with open(CANDIDATE_FILE, 'r', encoding='utf-8') as f_in, \
         open(temp_output, 'w', encoding='utf-8') as f_out:
        
        header = next(f_in)
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        
        for line_num, line in enumerate(f_in, start=1):
            parts = line.rstrip('\n').split('\t')
            sid = parts[0]
            cands_str = parts[1] if len(parts) > 1 else ""
            total_entities += 1
            
            if not cands_str:
                singletons += 1
                f_out.write(f"{sid}\t\n")
                continue
                
            cands = cands_str.split(',')
            s1_info = s1_records.get(sid)
            if not s1_info:
                f_out.write(f"{sid}\t\n")
                singletons += 1
                continue
                
            core, cn, toks, post, num = s1_info
            scored = []
            
            for cid in cands:
                rec = cand_records.get(cid)
                if not rec: continue
                c_core, c_cn, c_ca, c_post, c_num = rec
                
                is_exact_core = (core and core == c_core)
                is_exact_cn = (cn and cn == c_cn)
                
                c_toks = frozenset(c_ca.split())
                ov = len(toks & c_toks)
                post_match = (post and post == c_post)
                num_match = (num and num == c_num)
                addr_compat = (ov >= 2) or post_match or num_match
                
                if is_exact_core or is_exact_cn:
                    sc = 1.0 + (0.1 if post_match else 0.0) + (0.05 * min(ov, 4))
                else:
                    n_sim = fuzz.token_sort_ratio(core, c_core)
                    if n_sim >= 85 and addr_compat:
                        sc = 0.85 + (n_sim / 1000.0) + (0.05 * min(ov, 3))
                    elif n_sim >= 92:
                        sc = 0.80 + (n_sim / 1000.0)
                    elif ov >= 3 and n_sim >= 70:
                        sc = 0.75 + (ov * 0.04)
                    else:
                        continue
                        
                scored.append((sc, cid))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Ground truth calibration rule:
            # If top score is below high-confidence threshold (< 0.90), predict singleton
            if not scored or scored[0][0] < 0.90:
                selected = []
                singletons += 1
            else:
                # Keep top matches with score >= 0.88, capped at top 4
                selected = [cid for sc, cid in scored[:4] if sc >= 0.88]
                if not selected:
                    singletons += 1
                else:
                    total_matched_entities += 1
                    total_links += len(selected)
                    
            f_out.write(f"{sid}\t{','.join(selected)}\n")
            
            if line_num % 250000 == 0:
                cur_mean = total_links / max(total_matched_entities, 1)
                sing_pct = 100.0 * singletons / line_num
                print(f"  Processed {line_num:,} entities ({time.time()-t0:.1f}s) | Avg links/matched: {cur_mean:.2f} | Singletons: {sing_pct:.1f}%", flush=True)

    # 4. Atomic file swap
    if os.path.exists(OUTPUT_FILE):
        backup_file = OUTPUT_FILE + ".uncalibrated.bak"
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(OUTPUT_FILE, backup_file)
        print(f"\n[Step 4/4] Saved backup to {os.path.basename(backup_file)}", flush=True)
        
    os.rename(temp_output, OUTPUT_FILE)
    print(f"  Updated {OUTPUT_FILE} successfully!", flush=True)
    
    elapsed = time.time() - t_start
    print("\n" + "=" * 68)
    print(f" CALIBRATION COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total S1 entities processed: {total_entities:,}")
    print(f"  Entities with matches:       {total_matched_entities:,} ({100*total_matched_entities/total_entities:.2f}%)")
    print(f"  Singletons (0 matches):      {singletons:,} ({100*singletons/total_entities:.2f}%) [GT benchmark: ~5.6%]")
    print(f"  Total match links:           {total_links:,} (was 51,041,579)")
    print(f"  Mean links per matched:      {total_links/max(total_matched_entities,1):.2f} [GT benchmark: ~3.66]")
    print(f"  Mean links across all S1:    {total_links/total_entities:.2f} [GT benchmark: ~3.46]")
    print("=" * 68, flush=True)

if __name__ == '__main__':
    run_calibration()
