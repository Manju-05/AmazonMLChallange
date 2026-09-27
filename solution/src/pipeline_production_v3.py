"""
pipeline_production_v3.py - Optimized 1-pass S2/S3 read, all 7 blocking views.

KEY OPTIMIZATION vs v1/v2:
- Reads S2 and S3 exactly ONCE (not 3x, one per country)
- Builds all-country indexes simultaneously in a single pass
- Then processes each country partition from already-built indexes
- This cuts I/O by 3x, making the pipeline run in ~5 min instead of 20+ min

Other improvements:
- 6th View: Postal code (India 6-digit, US 5-digit, France 5-digit)
- 7th View: Bigram anchor (first 4-chars of first two tokens)
- All views ungated (always take union, no early stopping)
- Jaro-Winkler >= 0.92 matching rule
- Adaptive fuzzy threshold (85 for long names, 88 for short)
- Postal + name partial match rule
- Candidate cap 50 per view
"""
import sys, os, time, re, unicodedata, gc
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
    "private","limited","llc","incorporated","corporation","company",
    "pllc","llp","plc","pvt","ltd","inc","corp","co","sarl","sas",
    "sci","sa","eurl","snc","groupe","societe","gmbh","ag","bv","nv",
    "spa","srl","sl","ms","and","associates","enterprise","enterprises",
    "services","trading","solutions","technologies","tech","group",
    "international","india","national","global",
}
STOPWORDS = {
    "the","and","of","in","for","de","du","et","la","le",
    "des","les","en","d","l","au","aux","a",
}

RE_POSTAL_INDIA  = re.compile(r'\b([1-9][0-9]{5})\b')
RE_POSTAL_US     = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
RE_POSTAL_FRANCE = re.compile(r'\b([0-9]{5})\b')
RE_NUMBERS       = re.compile(r'\b\d+\b')

def clean_str(s):
    if not s: return ""
    s = unicodedata.normalize('NFKD', s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def get_core(cn):
    words = [w for w in cn.split() if w not in LEGAL_SUFFIXES and w not in STOPWORDS and len(w) >= 2]
    return " ".join(words) if words else cn

def compact_str(s): return re.sub(r'\s+', '', s)

def extract_addr_anchor(a):
    nums = RE_NUMBERS.findall(a)
    num = nums[0] if nums else ""
    toks = [t for t in a.split() if t.isalpha() and len(t) >= 4]
    return num, (toks[0] if toks else "")

def extract_postal(addr, country):
    if not addr: return ""
    if country == "India":   m = RE_POSTAL_INDIA.search(addr)
    elif country == "US":    m = RE_POSTAL_US.search(addr)
    elif country == "France":m = RE_POSTAL_FRANCE.search(addr)
    else: m = RE_POSTAL_INDIA.search(addr) or RE_POSTAL_US.search(addr)
    return m.group(1) if m else ""

def run_pipeline():
    t_total = time.time()
    print("="*70, flush=True)
    print(" AMAZON ML CHALLENGE 2026 - PIPELINE v3 (1-pass S2/S3 read)", flush=True)
    print("="*70, flush=True)

    # ── Phase 1: Read S1 ──
    print("\n[Phase 1] Reading S1 test entities...", flush=True)
    t0 = time.time()
    s1_by_country = defaultdict(list)
    s1_order = []
    with open(TEST_S1, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                sid, name, addr, country = p[0], p[1], p[2], p[3]
                s1_order.append(sid)
                s1_by_country[country].append((sid, name, addr))
    print(f"  Loaded {len(s1_order):,} S1 entities in {time.time()-t0:.2f}s", flush=True)
    for c, items in sorted(s1_by_country.items()):
        print(f"    {c}: {len(items):,}", flush=True)

    # ── Phase 2: Read S2+S3 ONCE, build per-country indexes simultaneously ──
    print("\n[Phase 2] Reading S2+S3 once, building per-country 7-view indexes...", flush=True)
    t0 = time.time()

    # Nested dict: country -> view_name -> {key: [cid,...]}
    # Also country -> cid -> (core, clean, addr, cmp)
    countries = set(s1_by_country.keys())

    # Per-country inverted indexes
    core_idx    = {c: defaultdict(list) for c in countries}
    clean_idx   = {c: defaultdict(list) for c in countries}
    compact_idx = {c: defaultdict(list) for c in countries}
    token2_idx  = {c: defaultdict(list) for c in countries}
    addr_idx    = {c: defaultdict(list) for c in countries}
    postal_idx  = {c: defaultdict(list) for c in countries}
    bigram_idx  = {c: defaultdict(list) for c in countries}
    cand_recs   = {c: {} for c in countries}

    count_by_country = defaultdict(int)
    total_cands = 0

    for path in [TEST_S2, TEST_S3]:
        src_name = os.path.basename(path)
        print(f"  Indexing {src_name}...", flush=True)
        t_src = time.time()
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4: continue
                cid, name, addr, country = p[0], p[1], p[2], p[3]
                if country not in countries: continue

                cn = clean_str(name)
                core = get_core(cn)
                ca = clean_str(addr)
                cmp = compact_str(core)

                cand_recs[country][cid] = (core, cn, ca, cmp)

                if core:  core_idx[country][core].append(cid)
                if cn:    clean_idx[country][cn].append(cid)
                if len(cmp) >= 6: compact_idx[country][cmp].append(cid)

                tokens = core.split()
                if len(tokens) >= 2:
                    token2_idx[country][(tokens[0], tokens[1])].append(cid)
                    bigram_idx[country][tokens[0][:4]+tokens[1][:4]].append(cid)
                elif len(tokens) == 1 and len(tokens[0]) >= 5:
                    token2_idx[country][(tokens[0], "")].append(cid)

                num, alpha = extract_addr_anchor(ca)
                if num and alpha: addr_idx[country][(num, alpha)].append(cid)

                postal = extract_postal(ca, country)
                if postal: postal_idx[country][postal].append(cid)

                count_by_country[country] += 1
                n += 1
        print(f"    {src_name}: {n:,} records in {time.time()-t_src:.2f}s", flush=True)
        total_cands += n

    print(f"\n  Total indexed: {total_cands:,} in {time.time()-t0:.2f}s", flush=True)
    for c in sorted(countries):
        print(f"    {c}: {count_by_country[c]:,} candidates, "
              f"core_keys={len(core_idx[c]):,}", flush=True)

    # ── Phase 3: Matching per country ──
    print("\n[Phase 3] Matching S1 entities against indexed candidates...", flush=True)
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    matched_results = {}
    candidate_results = {}

    for country, s1_items in sorted(s1_by_country.items()):
        t_c = time.time()
        print(f"\n  -- {country}: {len(s1_items):,} S1 entities --", flush=True)

        ci  = core_idx[country]
        cli = clean_idx[country]
        cpi = compact_idx[country]
        t2i = token2_idx[country]
        ai  = addr_idx[country]
        pi  = postal_idx[country]
        bi  = bigram_idx[country]
        cr  = cand_recs[country]

        c_matched = 0
        c_total_cands = 0

        for sid, name, addr in s1_items:
            cn = clean_str(name)
            core = get_core(cn)
            ca = clean_str(addr)
            cmp = compact_str(core)
            tokens = core.split()
            addr_toks = frozenset(ca.split())
            fuzzy_thresh = 85 if len(core) > 15 else 88
            postal = extract_postal(ca, country)

            cands = set()
            # View 1: Exact core
            if core in ci: cands.update(ci[core][:50])
            # View 2: Exact clean
            if cn in cli: cands.update(cli[cn][:50])
            # View 3: Compact core
            if cmp and cmp in cpi: cands.update(cpi[cmp][:30])
            # View 4: Token pair
            if len(tokens) >= 2:
                k = (tokens[0], tokens[1])
                if k in t2i: cands.update(t2i[k][:30])
            elif len(tokens) == 1 and len(tokens[0]) >= 5:
                k = (tokens[0], "")
                if k in t2i: cands.update(t2i[k][:30])
            # View 5: Address anchor
            if ca:
                num, alpha = extract_addr_anchor(ca)
                if num and alpha and (num, alpha) in ai:
                    cands.update(ai[(num, alpha)][:20])
            # View 6: Postal code
            if postal and postal in pi: cands.update(pi[postal][:50])
            # View 7: Bigram
            if len(tokens) >= 2:
                bg = tokens[0][:4]+tokens[1][:4]
                if bg in bi: cands.update(bi[bg][:30])

            matched = set()
            for cid in cands:
                c_core, c_clean, cand_addr, c_cmp = cr.get(cid, ("","","",""))
                # Rule A: Exact core
                if core and core == c_core: matched.add(cid); continue
                # Rule B: Exact clean
                if cn and cn == c_clean: matched.add(cid); continue
                # Rule C: Compact substring
                if cmp and c_cmp and len(cmp) >= 6 and len(c_cmp) >= 6:
                    if cmp in c_cmp or c_cmp in cmp: matched.add(cid); continue
                # Rule D: Fuzzy (adaptive)
                n_sim = fuzz.token_sort_ratio(core, c_core)
                if n_sim >= fuzzy_thresh: matched.add(cid); continue
                # Rule E: Address overlap
                if ca and cand_addr:
                    c_at = frozenset(cand_addr.split())
                    ov = len(addr_toks & c_at)
                    if ov >= 3 and n_sim >= 60: matched.add(cid); continue
                    if ov >= 4: matched.add(cid); continue
                # Rule F: Jaro-Winkler
                if core and c_core:
                    if JaroWinkler.normalized_similarity(core, c_core) >= 0.92:
                        matched.add(cid); continue
                # Rule G: Postal + name partial
                if postal and cand_addr:
                    cp = extract_postal(cand_addr, country)
                    if postal == cp and n_sim >= 55: matched.add(cid); continue

            cands.update(matched)
            matched_results[sid] = ",".join(sorted(matched)) if matched else ""
            candidate_results[sid] = ",".join(sorted(cands)) if cands else ""
            if matched: c_matched += 1
            c_total_cands += len(cands)

        mr = 100*c_matched/len(s1_items) if s1_items else 0
        ac = c_total_cands/len(s1_items) if s1_items else 0
        print(f"    Done in {time.time()-t_c:.2f}s | Match_rate={mr:.1f}% Avg_cands={ac:.1f}", flush=True)

        # Free this country's indexes from memory
        del ci, cli, cpi, t2i, ai, pi, bi, cr
        core_idx.pop(country, None)
        clean_idx.pop(country, None)
        compact_idx.pop(country, None)
        token2_idx.pop(country, None)
        addr_idx.pop(country, None)
        postal_idx.pop(country, None)
        bigram_idx.pop(country, None)
        cand_recs.pop(country, None)
        gc.collect()

    # ── Phase 4: Write outputs ──
    print("\n[Phase 4] Writing submission TSV files...", flush=True)
    t0 = time.time()
    with open(MATCHING_OUTPUT, "w", encoding="utf-8") as fm, \
         open(CANDIDATE_OUTPUT, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            fm.write(f"{sid}\t{matched_results.get(sid,'')}\n")
            fc.write(f"{sid}\t{candidate_results.get(sid,'')}\n")

    n_matched = sum(1 for v in matched_results.values() if v)
    elapsed = time.time() - t_total
    print(f"  Written {len(s1_order):,} rows in {time.time()-t0:.2f}s", flush=True)
    print(f"\n{'='*70}", flush=True)
    print(f" PIPELINE v3 COMPLETE in {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f" Matched: {n_matched:,} | Singletons: {len(s1_order)-n_matched:,}", flush=True)
    print(f" Output: {MATCHING_OUTPUT}", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    run_pipeline()
