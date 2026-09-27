"""
pipeline_production_v2.py - High-Performance Multilingual Business Entity Resolution.
UPGRADED VERSION with 7-view blocking and improved matching rules.

Improvements over v1:
1. All views are ALWAYS computed (no gating) - improves recall
2. 6th View: Postal code exact match (India PIN 6-digit, US ZIP 5-digit, France 5-digit)
3. 7th View: Name bigram pairs (additional anchor for partial name matches)
4. Rule F: Jaro-Winkler >= 0.92 (catches transpositions and typos)
5. Adaptive fuzzy threshold: 85 for long names (core len > 15), 88 otherwise
6. Compact substring threshold: lowered from 7 to 6 characters
7. Candidate cap raised from 30 to 50 per view for high-cardinality keys
8. Better postal code extraction (country-aware)
"""
import sys
import os
import time
import re
import unicodedata
import gc
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "student_resource", "student_resource", "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_S1 = os.path.join(DATASET_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(DATASET_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(DATASET_DIR, "test_source3.tsv")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

LEGAL_SUFFIXES = {
    "private", "limited", "llc", "incorporated", "corporation",
    "company", "pllc", "llp", "plc", "pvt", "ltd", "inc", "corp",
    "co", "sarl", "sas", "sci", "sa", "eurl", "snc", "groupe",
    "societe", "gmbh", "ag", "bv", "nv", "spa", "srl", "sl", "ms",
    "and", "associates", "enterprise", "enterprises", "services",
    "trading", "solutions", "technologies", "tech", "group",
    "international", "india", "national", "global",
}

STOPWORDS = {
    "the", "and", "of", "in", "for", "de", "du", "et", "la", "le",
    "des", "les", "en", "d", "l", "au", "aux", "a",
}

RE_POSTAL_INDIA = re.compile(r'\b([1-9][0-9]{5})\b')
RE_POSTAL_US = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
RE_POSTAL_FRANCE = re.compile(r'\b([0-9]{5})\b')
RE_NUMBERS = re.compile(r'\b\d+\b')


def clean_str(s):
    if not s:
        return ""
    s = unicodedata.normalize('NFKD', s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def get_core(clean_name):
    words = [w for w in clean_name.split()
             if w not in LEGAL_SUFFIXES and w not in STOPWORDS and len(w) >= 2]
    return " ".join(words) if words else clean_name


def compact_str(s):
    return re.sub(r'\s+', '', s)


def extract_addr_anchor(clean_addr):
    m_num = RE_NUMBERS.findall(clean_addr)
    num = m_num[0] if m_num else ""
    tokens = [t for t in clean_addr.split() if t.isalpha() and len(t) >= 4]
    alpha = tokens[0] if tokens else ""
    return num, alpha


def extract_postal(clean_addr, country):
    if not clean_addr:
        return ""
    if country == "India":
        m = RE_POSTAL_INDIA.search(clean_addr)
    elif country == "US":
        m = RE_POSTAL_US.search(clean_addr)
    elif country == "France":
        m = RE_POSTAL_FRANCE.search(clean_addr)
    else:
        m = RE_POSTAL_INDIA.search(clean_addr) or RE_POSTAL_US.search(clean_addr)
    return m.group(1) if m else ""


def run_pipeline():
    total_start = time.time()
    print("=" * 72, flush=True)
    print(" AMAZON ML CHALLENGE 2026 - PIPELINE v2 (7-view blocking)", flush=True)
    print("=" * 72, flush=True)

    print("\n[Phase 1] Ingesting test Source-1 entities...", flush=True)
    t0 = time.time()

    s1_by_country = defaultdict(list)
    s1_order = []

    with open(TEST_S1, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s1_id, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_order.append(s1_id)
                s1_by_country[country].append((s1_id, name, addr))

    print(f"  Loaded {len(s1_order):,} entities in {time.time() - t0:.2f}s", flush=True)
    for c, items in sorted(s1_by_country.items()):
        print(f"    - {c}: {len(items):,} entities", flush=True)

    matched_results = {}
    candidate_results = {}

    print("\n[Phase 2] Country-Partitioned 7-View Indexing & Resolution...", flush=True)

    for country, s1_items in sorted(s1_by_country.items()):
        country_start = time.time()
        print(f"\n  -- Processing Country: {country} ({len(s1_items):,} S1 entities) --", flush=True)

        core_index    = defaultdict(list)
        clean_index   = defaultdict(list)
        compact_index = defaultdict(list)
        token2_index  = defaultdict(list)
        addr_index    = defaultdict(list)
        postal_index  = defaultdict(list)
        bigram_index  = defaultdict(list)
        cand_records  = {}

        t_idx = time.time()
        c_count = 0

        for path in [TEST_S2, TEST_S3]:
            with open(path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        cid, name, addr = parts[0], parts[1], parts[2]
                        c_name = clean_str(name)
                        core = get_core(c_name)
                        c_addr = clean_str(addr)
                        cmp = compact_str(core)

                        cand_records[cid] = (core, c_name, c_addr, cmp)

                        if core:
                            core_index[core].append(cid)
                        if c_name:
                            clean_index[c_name].append(cid)
                        if len(cmp) >= 6:
                            compact_index[cmp].append(cid)

                        tokens = core.split()
                        if len(tokens) >= 2:
                            token2_index[(tokens[0], tokens[1])].append(cid)
                            bigram = tokens[0][:4] + tokens[1][:4]
                            bigram_index[bigram].append(cid)
                        elif len(tokens) == 1 and len(tokens[0]) >= 5:
                            token2_index[(tokens[0], "")].append(cid)

                        num, alpha = extract_addr_anchor(c_addr)
                        if num and alpha:
                            addr_index[(num, alpha)].append(cid)

                        postal = extract_postal(c_addr, country)
                        if postal:
                            postal_index[postal].append(cid)

                        c_count += 1

        print(f"    Indexed {c_count:,} candidates in {time.time() - t_idx:.2f}s", flush=True)
        print(f"      core={len(core_index):,} postal={len(postal_index):,} bigram={len(bigram_index):,}", flush=True)

        t_match = time.time()
        c_matched = 0
        c_candidates = 0

        from rapidfuzz import fuzz
        from rapidfuzz.distance import JaroWinkler

        for s1_id, name, addr in s1_items:
            c_name = clean_str(name)
            core = get_core(c_name)
            c_addr = clean_str(addr)
            cmp = compact_str(core)
            tokens = core.split()
            s1_addr_tokens = frozenset(c_addr.split())
            fuzzy_thresh = 85 if len(core) > 15 else 88
            postal = extract_postal(c_addr, country)

            cands = set()

            # View 1: Exact core
            if core in core_index:
                cands.update(core_index[core][:50])
            # View 2: Exact clean
            if c_name in clean_index:
                cands.update(clean_index[c_name][:50])
            # View 3: Compact core
            if cmp and cmp in compact_index:
                cands.update(compact_index[cmp][:30])
            # View 4: Token pair
            if len(tokens) >= 2:
                if (tokens[0], tokens[1]) in token2_index:
                    cands.update(token2_index[(tokens[0], tokens[1])][:30])
            elif len(tokens) == 1 and len(tokens[0]) >= 5:
                if (tokens[0], "") in token2_index:
                    cands.update(token2_index[(tokens[0], "")][:30])
            # View 5: Address anchor
            if c_addr:
                num, alpha = extract_addr_anchor(c_addr)
                if num and alpha and (num, alpha) in addr_index:
                    cands.update(addr_index[(num, alpha)][:20])
            # View 6: Postal code (NEW)
            if postal and postal in postal_index:
                cands.update(postal_index[postal][:50])
            # View 7: Bigram (NEW)
            if len(tokens) >= 2:
                bgram = tokens[0][:4] + tokens[1][:4]
                if bgram in bigram_index:
                    cands.update(bigram_index[bgram][:30])

            matched = set()

            for cid in cands:
                c_core, c_clean, cand_addr, c_cmp = cand_records.get(cid, ("", "", "", ""))

                # Rule A: Exact core
                if core and core == c_core:
                    matched.add(cid)
                    continue
                # Rule B: Exact clean
                if c_name and c_name == c_clean:
                    matched.add(cid)
                    continue
                # Rule C: Compact substring
                if cmp and c_cmp and len(cmp) >= 6 and len(c_cmp) >= 6:
                    if cmp in c_cmp or c_cmp in cmp:
                        matched.add(cid)
                        continue
                # Rule D: Fuzzy name (adaptive threshold)
                n_sim = fuzz.token_sort_ratio(core, c_core)
                if n_sim >= fuzzy_thresh:
                    matched.add(cid)
                    continue
                # Rule E: Address overlap + partial name
                if c_addr and cand_addr:
                    c_addr_toks = frozenset(cand_addr.split())
                    addr_overlap = len(s1_addr_tokens & c_addr_toks)
                    if addr_overlap >= 3 and n_sim >= 60:
                        matched.add(cid)
                        continue
                    elif addr_overlap >= 4:
                        matched.add(cid)
                        continue
                # Rule F: Jaro-Winkler (NEW)
                if core and c_core:
                    jw = JaroWinkler.normalized_similarity(core, c_core)
                    if jw >= 0.92:
                        matched.add(cid)
                        continue
                # Rule G: Postal match + name similarity (NEW)
                if postal and cand_addr:
                    c_postal = extract_postal(cand_addr, country)
                    if postal == c_postal and n_sim >= 55:
                        matched.add(cid)
                        continue

            cands.update(matched)

            matched_results[s1_id] = ",".join(sorted(matched)) if matched else ""
            candidate_results[s1_id] = ",".join(sorted(cands)) if cands else ""

            if matched:
                c_matched += 1
            c_candidates += len(cands)

        mr = 100 * c_matched / len(s1_items) if s1_items else 0
        ac = c_candidates / len(s1_items) if s1_items else 0
        print(f"    Matched {len(s1_items):,} in {time.time() - t_match:.2f}s | "
              f"Rate={mr:.1f}% Avg_cands={ac:.1f}", flush=True)

        del core_index, clean_index, compact_index, token2_index
        del addr_index, postal_index, bigram_index, cand_records
        gc.collect()

    print("\n[Phase 3] Writing output TSV files...", flush=True)
    t_write = time.time()

    with open(MATCHING_OUTPUT, "w", encoding="utf-8") as f_m, \
         open(CANDIDATE_OUTPUT, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_order:
            f_m.write(f"{s1_id}\t{matched_results.get(s1_id, '')}\n")
            f_c.write(f"{s1_id}\t{candidate_results.get(s1_id, '')}\n")

    matched_count = sum(1 for v in matched_results.values() if v)
    print(f"  Written {len(s1_order):,} rows in {time.time() - t_write:.2f}s", flush=True)
    elapsed = time.time() - total_start
    print(f"\n  Total: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"  Matched: {matched_count:,} | Singletons: {len(s1_order)-matched_count:,}", flush=True)


if __name__ == "__main__":
    run_pipeline()
