import pandas as pd
import numpy as np
import time, re, unicodedata
from rapidfuzz import fuzz
from collections import defaultdict

t0 = time.time()
print("Starting F0.5 benchmark on training sample...")

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

def clean_name(s):
    if not s: return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def get_core(cn):
    words = [w for w in cn.split() if w not in LEGAL and w not in STOPWORDS and len(w) >= 2]
    return " ".join(words) if words else cn

def f05(pred, actual):
    if not actual and not pred: return 1.0
    if not actual or not pred: return 0.0
    tp = len(pred & actual)
    fp = len(pred - actual)
    fn = len(actual - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0: return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)

# Load 2,000 S1 training entities and ground truth
s1_df = pd.read_csv('student_resource/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=2000)
gt_df = pd.read_csv('student_resource/student_resource/dataset/train/train_ground_truth.tsv', sep='\t').set_index('source1_entity_id')

gt = {}
target_cids = set()
for sid in s1_df['entity_id']:
    if sid in gt_df.index:
        m = str(gt_df.loc[sid, 'matched_entity_ids'])
        mids = set(m.split(',')) if m and m != 'nan' else set()
        gt[sid] = mids
        target_cids.update(mids)
    else:
        gt[sid] = set()

print(f"Loaded {len(s1_df)} S1 entities, {len(target_cids)} true match targets.")

# Add some negative distractor candidates to test precision calibration
# Load sample from S2 and S3
s2_df = pd.read_csv('student_resource/student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=50000)
s3_df = pd.read_csv('student_resource/student_resource/dataset/train/train_source3.tsv', sep='\t', nrows=50000)

cand_recs = {}
for df in [s2_df, s3_df]:
    for _, row in df.iterrows():
        cid = row['entity_id']
        cn = clean_name(str(row['business_name']))
        ca = clean_name(str(row['business_address']))
        country = str(row['country'])
        core = get_core(cn)
        toks = frozenset(ca.split())
        post = get_postal(ca, country)
        num = get_anchor_num(ca)
        cand_recs[cid] = (core, toks, post, num)

# Also load any target_cids not in first 50k
missing_targets = target_cids - set(cand_recs.keys())
print(f"Retrieving {len(missing_targets)} remaining target records...")
for path in ['train_source2.tsv', 'train_source3.tsv']:
    with open('student_resource/student_resource/dataset/train/' + path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in missing_targets:
                cid, name, addr, country = p[0], p[1], p[2], p[3]
                cn = clean_name(name)
                ca = clean_name(addr)
                core = get_core(cn)
                toks = frozenset(ca.split())
                post = get_postal(ca, country)
                num = get_anchor_num(ca)
                cand_recs[cid] = (core, toks, post, num)

print(f"Total candidate pool: {len(cand_recs)} records.")

# For each S1 entity, form a candidate pool containing:
# 1) all true targets (simulating perfect recall blocking)
# 2) ~35 negative candidates (simulating real blocking output of ~40 candidates)
# Then test our scoring and ranking model!
import random
random.seed(42)
all_cand_ids = list(cand_recs.keys())

scores_raw = []
scores_calibrated = []

for _, row in s1_df.iterrows():
    sid = row['entity_id']
    country = str(row['country'])
    cn = clean_name(str(row['business_name']))
    ca = clean_name(str(row['business_address']))
    core = get_core(cn)
    toks = frozenset(ca.split())
    post = get_postal(ca, country)
    num = get_anchor_num(ca)
    
    true_m = gt[sid]
    # Simulate candidate pool: true matches + 35 random candidates
    cands = set(true_m)
    cands.update(random.sample(all_cand_ids, min(35, len(all_cand_ids))))
    
    # 1. Uncalibrated logic (what pipeline_turbo was doing):
    raw_pred = set()
    for cid in cands:
        c_core, c_toks, c_post, c_num = cand_recs.get(cid, ("", frozenset(), "", ""))
        if core and core == c_core: raw_pred.add(cid); continue
        if len(core) >= 6 and len(c_core) >= 6 and (core in c_core or c_core in core):
            raw_pred.add(cid); continue
        if fuzz.token_sort_ratio(core, c_core) >= 80: raw_pred.add(cid); continue
    scores_raw.append(f05(raw_pred, true_m))
    
    # 2. Calibrated logic (Option B):
    scored_cands = []
    for cid in cands:
        c_core, c_toks, c_post, c_num = cand_recs.get(cid, ("", frozenset(), "", ""))
        if not c_core: continue
        
        # Name score
        is_exact = (core and core == c_core)
        n_sim = 100.0 if is_exact else fuzz.token_sort_ratio(core, c_core)
        
        # Address compatibility
        ov = len(toks & c_toks)
        post_match = (post and post == c_post)
        num_match = (num and num == c_num)
        addr_compat = (ov >= 2) or post_match or num_match
        
        # Combined score
        if is_exact:
            sc = 1.0 + (0.1 if addr_compat else 0.0) + (0.05 * min(ov, 4))
        elif n_sim >= 85 and addr_compat:
            sc = 0.85 + (n_sim / 1000.0) + (0.05 * min(ov, 3))
        elif n_sim >= 92:
            sc = 0.80 + (n_sim / 1000.0)
        elif ov >= 3 and n_sim >= 70:
            sc = 0.75 + (ov * 0.05)
        else:
            continue
            
        scored_cands.append((sc, cid))
    
    # Sort descending by score
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    # Keep top 4-5 with score >= 0.80
    calib_pred = {cid for sc, cid in scored_cands[:5] if sc >= 0.80}
    scores_calibrated.append(f05(calib_pred, true_m))

print("\n" + "="*60)
print(f"BENCHMARK RESULTS (N={len(s1_df)} S1 entities):")
print(f"  Uncalibrated Macro F0.5 : {np.mean(scores_raw):.4f}")
print(f"  Calibrated Macro F0.5   : {np.mean(scores_calibrated):.4f}")
print(f"  Improvement             : +{(np.mean(scores_calibrated) - np.mean(scores_raw)):.4f} ({100*(np.mean(scores_calibrated)/max(np.mean(scores_raw),1e-5)-1):.1f}%)")
print("="*60)
