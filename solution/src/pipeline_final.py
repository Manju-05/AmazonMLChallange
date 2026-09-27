"""
pipeline_final.py - FASTEST production pipeline using pandas vectorized ops.

Key speed optimizations:
- pandas read_csv with chunked processing (C-level TSV parsing, 10x faster than line-by-line)
- Vectorized string normalization via pandas .str methods (avoids Python loop per row)
- 1-pass S2/S3 read across all countries
- 7-view inverted index blocking
- All matching rules from v3
"""
import sys, os, time, re, unicodedata, gc
from collections import defaultdict
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR),
    "student_resource", "student_resource", "dataset", "test")
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
STOPWORDS = {"the","and","of","in","for","de","du","et","la","le",
             "des","les","en","d","l","au","aux","a"}
LS_SET = LEGAL_SUFFIXES | STOPWORDS

RE_NON_ALNUM = re.compile(r'[^a-z0-9\s]')
RE_MULTI_SP  = re.compile(r'\s+')
RE_POSTAL_IN = re.compile(r'(?<!\d)([1-9]\d{5})(?!\d)')
RE_POSTAL_US = re.compile(r'(?<!\d)(\d{5})(?:-\d{4})?(?!\d)')
RE_POSTAL_FR = re.compile(r'(?<!\d)(\d{5})(?!\d)')
RE_NUMBERS   = re.compile(r'\b\d+\b')


def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorized: lowercase + NFKD accent strip + keep alphanumeric+space."""
    # 1. Fill nulls
    s = s.fillna("").astype(str)
    # 2. Apply NFKD + accent strip (must be row-level but fast via map)
    def _nfkd(x):
        x = unicodedata.normalize('NFKD', x)
        return "".join(c for c in x if not unicodedata.combining(c))
    s = s.map(_nfkd)
    # 3. Vectorized: lowercase, strip non-alnum, collapse spaces
    s = s.str.lower()
    s = s.str.replace(RE_NON_ALNUM, ' ', regex=True)
    s = s.str.replace(RE_MULTI_SP, ' ', regex=True)
    return s.str.strip()


def get_core_series(clean: pd.Series) -> pd.Series:
    """Vectorized core name extraction - strip legal suffixes & stopwords."""
    def _core(name):
        words = [w for w in name.split()
                 if w not in LS_SET and len(w) >= 2]
        return " ".join(words) if words else name
    return clean.map(_core)


def compact_series(s: pd.Series) -> pd.Series:
    return s.str.replace(r'\s+', '', regex=True)


def extract_postal_series(addr: pd.Series, country: str) -> pd.Series:
    def _postal(a):
        if not a: return ""
        if country == "India":
            m = RE_POSTAL_IN.search(a)
        elif country == "US":
            m = RE_POSTAL_US.search(a)
        elif country == "France":
            m = RE_POSTAL_FR.search(a)
        else:
            m = RE_POSTAL_IN.search(a) or RE_POSTAL_US.search(a)
        return m.group(1) if m else ""
    return addr.map(_postal)


def extract_addr_anchor(a):
    if not a: return "", ""
    nums = RE_NUMBERS.findall(a)
    toks = [t for t in a.split() if t.isalpha() and len(t) >= 4]
    return (nums[0] if nums else ""), (toks[0] if toks else "")


def load_and_preprocess(path, countries_filter=None):
    """Load TSV and preprocess with pandas vectorized ops."""
    print(f"  Loading {os.path.basename(path)}...", flush=True)
    t = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str,
                     usecols=['entity_id','business_name','business_address','country'],
                     low_memory=False)
    df = df.fillna("")
    if countries_filter:
        df = df[df['country'].isin(countries_filter)].reset_index(drop=True)
    print(f"    {len(df):,} records loaded in {time.time()-t:.1f}s", flush=True)

    print(f"    Normalizing names...", flush=True)
    t = time.time()
    df['name_clean'] = normalize_series(df['business_name'])
    df['name_core']  = get_core_series(df['name_clean'])
    df['name_cmp']   = compact_series(df['name_core'])
    print(f"    Names done in {time.time()-t:.1f}s", flush=True)

    print(f"    Normalizing addresses...", flush=True)
    t = time.time()
    df['addr_clean'] = normalize_series(df['business_address'])
    print(f"    Addresses done in {time.time()-t:.1f}s", flush=True)

    return df


def build_indexes(df, countries):
    """Build per-country 7-view inverted indexes from a candidate DataFrame."""
    ci  = {c: defaultdict(list) for c in countries}
    cli = {c: defaultdict(list) for c in countries}
    cpi = {c: defaultdict(list) for c in countries}
    t2i = {c: defaultdict(list) for c in countries}
    ai  = {c: defaultdict(list) for c in countries}
    pi  = {c: defaultdict(list) for c in countries}
    bi  = {c: defaultdict(list) for c in countries}
    cr  = {c: {} for c in countries}

    print(f"    Building indexes for {len(df):,} records...", flush=True)
    t = time.time()

    # Add postal per country (country-aware regex)
    for country in countries:
        sub = df[df['country'] == country]
        postals = extract_postal_series(sub['addr_clean'], country)
        df.loc[sub.index, 'postal'] = postals

    df['postal'] = df.get('postal', pd.Series("", index=df.index)).fillna("")

    rows = df[['entity_id','name_core','name_clean','addr_clean','name_cmp','postal','country']].values

    for row in rows:
        cid, core, cn, ca, cmp, postal, country = row
        if country not in countries: continue

        cr[country][cid] = (core, cn, ca, cmp)

        if core:  ci[country][core].append(cid)
        if cn:    cli[country][cn].append(cid)
        if len(cmp) >= 6: cpi[country][cmp].append(cid)

        tokens = core.split()
        if len(tokens) >= 2:
            t2i[country][(tokens[0], tokens[1])].append(cid)
            bi[country][tokens[0][:4]+tokens[1][:4]].append(cid)
        elif len(tokens) == 1 and len(tokens[0]) >= 5:
            t2i[country][(tokens[0], "")].append(cid)

        num, alpha = extract_addr_anchor(ca)
        if num and alpha: ai[country][(num, alpha)].append(cid)
        if postal: pi[country][postal].append(cid)

    print(f"    Indexes built in {time.time()-t:.1f}s", flush=True)
    return ci, cli, cpi, t2i, ai, pi, bi, cr


def run_pipeline():
    t_total = time.time()
    print("="*70, flush=True)
    print(" AMAZON ML CHALLENGE 2026 - FINAL PIPELINE (pandas vectorized)", flush=True)
    print("="*70, flush=True)

    # ── Phase 1: Load S1 ──
    print("\n[Phase 1] Loading & preprocessing S1 test data...", flush=True)
    s1_df = load_and_preprocess(TEST_S1)
    s1_order = s1_df['entity_id'].tolist()
    countries = set(s1_df['country'].unique())
    s1_by_country = {c: s1_df[s1_df['country']==c] for c in countries}
    print(f"  Countries: {sorted(countries)}", flush=True)
    for c in sorted(countries):
        print(f"    {c}: {len(s1_by_country[c]):,}", flush=True)

    # ── Phase 2: Load & index S2 + S3 ──
    print("\n[Phase 2] Loading & indexing S2+S3 (1 pass each)...", flush=True)
    s2_df = load_and_preprocess(TEST_S2, countries_filter=countries)
    s3_df = load_and_preprocess(TEST_S3, countries_filter=countries)
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df; gc.collect()

    print(f"\n  Combined S2+S3: {len(cand_df):,} records", flush=True)
    ci, cli, cpi, t2i, ai, pi, bi, cr = build_indexes(cand_df, countries)
    del cand_df; gc.collect()

    # ── Phase 3: Matching ──
    print("\n[Phase 3] Matching per country...", flush=True)
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    matched_results = {}
    candidate_results = {}

    for country in sorted(countries):
        sub = s1_by_country[country]
        print(f"\n  -- {country}: {len(sub):,} S1 entities --", flush=True)
        t_c = time.time()

        # Add postal for S1 in this country
        postals = extract_postal_series(sub['addr_clean'], country)

        c_matched = 0
        c_total_cands = 0

        # Unpack indexes for this country
        _ci=ci[country]; _cli=cli[country]; _cpi=cpi[country]
        _t2i=t2i[country]; _ai=ai[country]; _pi=pi[country]
        _bi=bi[country]; _cr=cr[country]

        for (sid, core, cn, ca, cmp), postal in zip(
            sub[['entity_id','name_core','name_clean','addr_clean','name_cmp']].values,
            postals.values
        ):
            tokens = core.split()
            addr_toks = frozenset(ca.split())
            fuzzy_thresh = 85 if len(core) > 15 else 88

            cands = set()
            # View 1: Exact core
            if core in _ci: cands.update(_ci[core][:50])
            # View 2: Exact clean
            if cn in _cli: cands.update(_cli[cn][:50])
            # View 3: Compact core
            if cmp and cmp in _cpi: cands.update(_cpi[cmp][:30])
            # View 4: Token pair
            if len(tokens) >= 2:
                k=(tokens[0],tokens[1])
                if k in _t2i: cands.update(_t2i[k][:30])
            elif len(tokens)==1 and len(tokens[0])>=5:
                k=(tokens[0],"")
                if k in _t2i: cands.update(_t2i[k][:30])
            # View 5: Address anchor
            if ca:
                num, alpha = extract_addr_anchor(ca)
                if num and alpha and (num,alpha) in _ai:
                    cands.update(_ai[(num,alpha)][:20])
            # View 6: Postal code
            if postal and postal in _pi:
                cands.update(_pi[postal][:50])
            # View 7: Bigram
            if len(tokens) >= 2:
                bg = tokens[0][:4]+tokens[1][:4]
                if bg in _bi: cands.update(_bi[bg][:30])

            matched = set()
            for cid in cands:
                c_core,c_clean,cand_addr,c_cmp = _cr.get(cid,("","","",""))
                if core and core==c_core: matched.add(cid); continue
                if cn and cn==c_clean: matched.add(cid); continue
                if cmp and c_cmp and len(cmp)>=6 and len(c_cmp)>=6:
                    if cmp in c_cmp or c_cmp in cmp: matched.add(cid); continue
                n_sim = fuzz.token_sort_ratio(core, c_core)
                if n_sim >= fuzzy_thresh: matched.add(cid); continue
                if ca and cand_addr:
                    c_at=frozenset(cand_addr.split())
                    ov=len(addr_toks & c_at)
                    if ov>=3 and n_sim>=60: matched.add(cid); continue
                    if ov>=4: matched.add(cid); continue
                if core and c_core:
                    if JaroWinkler.normalized_similarity(core,c_core)>=0.92:
                        matched.add(cid); continue
                if postal and cand_addr:
                    cp_re = (RE_POSTAL_IN if country=="India" else
                             RE_POSTAL_US if country=="US" else RE_POSTAL_FR)
                    m2 = cp_re.search(cand_addr)
                    cp = m2.group(1) if m2 else ""
                    if postal==cp and n_sim>=55: matched.add(cid); continue

            cands.update(matched)
            matched_results[sid] = ",".join(sorted(matched)) if matched else ""
            candidate_results[sid] = ",".join(sorted(cands)) if cands else ""
            if matched: c_matched += 1
            c_total_cands += len(cands)

        mr = 100*c_matched/len(sub) if len(sub) else 0
        ac = c_total_cands/len(sub) if len(sub) else 0
        print(f"    Done in {time.time()-t_c:.1f}s | Match={mr:.1f}% AvgCands={ac:.1f}", flush=True)

    # ── Phase 4: Write outputs ──
    print("\n[Phase 4] Writing submission files...", flush=True)
    t0 = time.time()
    with open(MATCHING_OUTPUT,"w",encoding="utf-8") as fm, \
         open(CANDIDATE_OUTPUT,"w",encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            fm.write(f"{sid}\t{matched_results.get(sid,'')}\n")
            fc.write(f"{sid}\t{candidate_results.get(sid,'')}\n")

    n_matched = sum(1 for v in matched_results.values() if v)
    elapsed = time.time()-t_total
    print(f"\n{'='*70}", flush=True)
    print(f" DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f" Matched: {n_matched:,} | Singletons: {len(s1_order)-n_matched:,}", flush=True)
    print(f" -> {MATCHING_OUTPUT}", flush=True)
    print(f" -> {CANDIDATE_OUTPUT}", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    run_pipeline()
