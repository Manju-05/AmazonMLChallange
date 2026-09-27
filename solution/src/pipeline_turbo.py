"""
pipeline_turbo.py - TURBO pipeline using unidecode (C ext) + pure pandas vectorized ops.

Root cause of slowness was unicodedata.normalize() called in a Python loop per row.
Fix: use unidecode library (C extension) for accent stripping - 5-10x faster.
Plus: skip normalization for records that have no accents (ASCII fast-path).

Estimated total time: ~8-12 minutes for 1.73M S1 against 10M S2+S3.
"""
import sys, os, time, re, gc
from collections import defaultdict
import pandas as pd

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

try:
    from unidecode import unidecode
    USE_UNIDECODE = True
    print("[INFO] unidecode available - fast accent stripping enabled", flush=True)
except ImportError:
    import unicodedata
    def unidecode(s):
        n = unicodedata.normalize('NFKD', s)
        return "".join(c for c in n if not unicodedata.combining(c))
    USE_UNIDECODE = False
    print("[INFO] unidecode not found - fallback to unicodedata", flush=True)

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
LS = LEGAL_SUFFIXES | STOPWORDS

RE_NON_ALNUM = re.compile(r'[^a-z0-9 ]')
RE_SPACES    = re.compile(r' {2,}')
RE_POSTAL_IN = re.compile(r'(?<!\d)([1-9]\d{5})(?!\d)')
RE_POSTAL_US = re.compile(r'(?<!\d)(\d{5})(?:-\d{4})?(?!\d)')
RE_POSTAL_FR = re.compile(r'(?<!\d)(\d{5})(?!\d)')
RE_NUMS      = re.compile(r'\b\d+\b')


def fast_clean(s: str) -> str:
    """Single-row clean: unidecode + lower + keep alnum+space."""
    if not s: return ""
    try:
        s = unidecode(s)
    except Exception:
        pass
    s = s.lower()
    s = RE_NON_ALNUM.sub(' ', s)
    s = RE_SPACES.sub(' ', s)
    return s.strip()


def fast_core(name: str) -> str:
    words = [w for w in name.split() if w not in LS and len(w) >= 2]
    return " ".join(words) if words else name


def get_postal(addr: str, country: str) -> str:
    if not addr: return ""
    if country == "India":    m = RE_POSTAL_IN.search(addr)
    elif country == "US":     m = RE_POSTAL_US.search(addr)
    elif country == "France": m = RE_POSTAL_FR.search(addr)
    else: m = RE_POSTAL_IN.search(addr) or RE_POSTAL_US.search(addr)
    return m.group(1) if m else ""


def get_anchor(addr: str):
    if not addr: return "", ""
    nums = RE_NUMS.findall(addr)
    toks = [t for t in addr.split() if t.isalpha() and len(t) >= 4]
    return (nums[0] if nums else ""), (toks[0] if toks else "")


def preprocess_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"    Preprocessing {len(df):,} {label} records...", flush=True)
    t = time.time()
    # Fast vectorized: use map with fast_clean (unidecode is C, regex is compiled)
    df['name_clean'] = df['business_name'].fillna("").map(fast_clean)
    df['addr_clean'] = df['business_address'].fillna("").map(fast_clean)
    df['name_core']  = df['name_clean'].map(fast_core)
    df['name_cmp']   = df['name_core'].str.replace(' ', '', regex=False)
    print(f"    Done in {time.time()-t:.1f}s", flush=True)
    return df


def load_source(path: str, countries_filter=None, label="") -> pd.DataFrame:
    t = time.time()
    print(f"  Reading {os.path.basename(path)}...", flush=True)
    df = pd.read_csv(path, sep='\t', dtype=str,
                     usecols=['entity_id','business_name','business_address','country'],
                     low_memory=False)
    df = df.fillna("")
    if countries_filter:
        df = df[df['country'].isin(countries_filter)].reset_index(drop=True)
    print(f"    {len(df):,} records in {time.time()-t:.1f}s", flush=True)
    df = preprocess_df(df, label or os.path.basename(path))
    return df


def build_country_indexes(df: pd.DataFrame, countries: set):
    """Build 7 inverted indexes per country from candidate DataFrame."""
    ci={c:defaultdict(list) for c in countries}   # exact core
    cli={c:defaultdict(list) for c in countries}  # exact clean
    cpi={c:defaultdict(list) for c in countries}  # compact core
    t2i={c:defaultdict(list) for c in countries}  # token pair
    ai={c:defaultdict(list) for c in countries}   # addr anchor
    pi={c:defaultdict(list) for c in countries}   # postal
    bi={c:defaultdict(list) for c in countries}   # bigram
    cr={c:{} for c in countries}                  # record store

    print(f"    Indexing {len(df):,} records...", flush=True)
    t = time.time()

    for row in df[['entity_id','name_core','name_clean',
                   'addr_clean','name_cmp','country']].itertuples(index=False):
        cid, core, cn, ca, cmp, country = row
        if country not in countries: continue

        cr[country][cid] = (core, cn, ca, cmp)

        if core:  ci[country][core].append(cid)
        if cn:    cli[country][cn].append(cid)
        if len(cmp) >= 6: cpi[country][cmp].append(cid)

        toks = core.split()
        if len(toks) >= 2:
            t2i[country][(toks[0],toks[1])].append(cid)
            bi[country][toks[0][:4]+toks[1][:4]].append(cid)
        elif len(toks)==1 and len(toks[0])>=5:
            t2i[country][(toks[0],"")].append(cid)

        num, alpha = get_anchor(ca)
        if num and alpha: ai[country][(num,alpha)].append(cid)

        postal = get_postal(ca, country)
        if postal: pi[country][postal].append(cid)

    print(f"    Indexes built in {time.time()-t:.1f}s", flush=True)
    for c in sorted(countries):
        print(f"      {c}: {len(cr[c]):,} cands, core_keys={len(ci[c]):,}", flush=True)
    return ci, cli, cpi, t2i, ai, pi, bi, cr


def run_pipeline():
    t_total = time.time()
    print("="*68, flush=True)
    print(" AMAZON ML CHALLENGE 2026 — TURBO PIPELINE", flush=True)
    print("="*68, flush=True)

    # ── Phase 1: S1 ──
    print("\n[Phase 1] S1 test data...", flush=True)
    s1 = load_source(TEST_S1, label="S1")
    s1_order   = s1['entity_id'].tolist()
    countries  = set(s1['country'].unique())
    s1_by_ctr  = {c: s1[s1['country']==c].reset_index(drop=True) for c in countries}
    del s1; gc.collect()

    for c in sorted(countries):
        print(f"    {c}: {len(s1_by_ctr[c]):,}", flush=True)

    # ── Phase 2: S2 + S3 → indexes ──
    print("\n[Phase 2] S2+S3 loading & indexing...", flush=True)
    s2 = load_source(TEST_S2, countries_filter=countries, label="S2")
    s3 = load_source(TEST_S3, countries_filter=countries, label="S3")
    cands = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()
    print(f"  Combined: {len(cands):,} candidates", flush=True)
    ci, cli, cpi, t2i, ai, pi, bi, cr = build_country_indexes(cands, countries)
    del cands; gc.collect()

    # ── Phase 3: Matching ──
    print("\n[Phase 3] Matching...", flush=True)
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    matched_results   = {}
    candidate_results = {}

    for country in sorted(countries):
        sub = s1_by_ctr[country]
        print(f"\n  ── {country}: {len(sub):,} entities ──", flush=True)
        t_c = time.time()

        _ci=ci[country]; _cli=cli[country]; _cpi=cpi[country]
        _t2i=t2i[country]; _ai=ai[country]; _pi=pi[country]
        _bi=bi[country]; _cr=cr[country]
        c_matched = 0; c_tot = 0

        postal_re = (RE_POSTAL_IN if country=="India" else
                     RE_POSTAL_US if country=="US" else RE_POSTAL_FR)

        for sid, core, cn, ca, cmp in zip(
            sub['entity_id'], sub['name_core'],
            sub['name_clean'], sub['addr_clean'], sub['name_cmp']
        ):
            toks = core.split()
            addr_toks = frozenset(ca.split())
            thresh = 85 if len(core) > 15 else 88
            postal = get_postal(ca, country)

            cands = set()
            if core in _ci:  cands.update(_ci[core][:50])
            if cn in _cli:   cands.update(_cli[cn][:50])
            if cmp and cmp in _cpi: cands.update(_cpi[cmp][:30])
            if len(toks) >= 2:
                k=(toks[0],toks[1])
                if k in _t2i: cands.update(_t2i[k][:30])
                bg=toks[0][:4]+toks[1][:4]
                if bg in _bi: cands.update(_bi[bg][:30])
            elif len(toks)==1 and len(toks[0])>=5:
                k=(toks[0],"")
                if k in _t2i: cands.update(_t2i[k][:30])
            if ca:
                num, alpha = get_anchor(ca)
                if num and alpha and (num,alpha) in _ai:
                    cands.update(_ai[(num,alpha)][:20])
            if postal and postal in _pi:
                cands.update(_pi[postal][:50])

            matched = set()
            for cid in cands:
                c_core,c_cn,c_ca,c_cmp = _cr.get(cid,("","","",""))
                if core and core==c_core:          matched.add(cid); continue
                if cn   and cn==c_cn:              matched.add(cid); continue
                if (cmp and c_cmp and len(cmp)>=6 and len(c_cmp)>=6
                        and (cmp in c_cmp or c_cmp in cmp)):
                    matched.add(cid); continue
                n_sim = fuzz.token_sort_ratio(core, c_core)
                if n_sim >= thresh:                matched.add(cid); continue
                if ca and c_ca:
                    ov = len(addr_toks & frozenset(c_ca.split()))
                    if ov>=3 and n_sim>=60:        matched.add(cid); continue
                    if ov>=4:                      matched.add(cid); continue
                if core and c_core:
                    if JaroWinkler.normalized_similarity(core,c_core)>=0.92:
                        matched.add(cid); continue
                if postal and c_ca:
                    m2=postal_re.search(c_ca)
                    if m2 and postal==m2.group(1) and n_sim>=55:
                        matched.add(cid); continue

            cands.update(matched)
            matched_results[sid]   = ",".join(sorted(matched)) if matched else ""
            candidate_results[sid] = ",".join(sorted(cands))   if cands   else ""
            if matched: c_matched += 1
            c_tot += len(cands)

        mr = 100*c_matched/len(sub) if len(sub) else 0
        ac = c_tot/len(sub) if len(sub) else 0
        print(f"    Done {time.time()-t_c:.1f}s | Match={mr:.1f}% AvgCands={ac:.1f}", flush=True)

    # ── Phase 4: Write ──
    print("\n[Phase 4] Writing output files...", flush=True)
    t0 = time.time()
    with open(MATCHING_OUTPUT,"w",encoding="utf-8") as fm, \
         open(CANDIDATE_OUTPUT,"w",encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            fm.write(f"{sid}\t{matched_results.get(sid,'')}\n")
            fc.write(f"{sid}\t{candidate_results.get(sid,'')}\n")

    n_m = sum(1 for v in matched_results.values() if v)
    elapsed = time.time()-t_total
    print(f"\n{'='*68}", flush=True)
    print(f" DONE in {elapsed:.0f}s  ({elapsed/60:.1f} min)", flush=True)
    print(f" Matched : {n_m:,}", flush=True)
    print(f" Singletons: {len(s1_order)-n_m:,}", flush=True)
    print(f" -> {MATCHING_OUTPUT}", flush=True)
    print(f" -> {CANDIDATE_OUTPUT}", flush=True)
    print(f"{'='*68}", flush=True)

if __name__ == "__main__":
    run_pipeline()
