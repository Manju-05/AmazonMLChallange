"""
local_eval_v3.py - Fast F0.5 evaluation of v3 pipeline logic on 5000 training entities.
Uses the same 1-pass S2/S3 read optimization as pipeline_production_v3.py
"""
import sys, os, re, unicodedata, gc, time
from collections import defaultdict

BASE = os.path.abspath('.')
TRAIN_S1 = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_source1.tsv')
TRAIN_S2 = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_source2.tsv')
TRAIN_S3 = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_source3.tsv')
TRAIN_GT = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_ground_truth.tsv')

N_SAMPLE = 5000

LEGAL_SUFFIXES = {
    "private","limited","llc","incorporated","corporation","company",
    "pllc","llp","plc","pvt","ltd","inc","corp","co","sarl","sas",
    "sci","sa","eurl","snc","groupe","societe","gmbh","ag","bv","nv",
    "spa","srl","sl","ms","and","associates","enterprise","enterprises",
    "services","trading","solutions","technologies","tech","group",
    "international","india","national","global",
}
STOPWORDS = {"the","and","of","in","for","de","du","et","la","le","des","les","en","d","l","au","aux","a"}

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
    toks = [t for t in a.split() if t.isalpha() and len(t) >= 4]
    return (nums[0] if nums else ""), (toks[0] if toks else "")

def extract_postal(addr, country):
    if not addr: return ""
    if country == "India":   m = RE_POSTAL_INDIA.search(addr)
    elif country == "US":    m = RE_POSTAL_US.search(addr)
    elif country == "France":m = RE_POSTAL_FRANCE.search(addr)
    else: m = RE_POSTAL_INDIA.search(addr) or RE_POSTAL_US.search(addr)
    return m.group(1) if m else ""

def f05(pred, actual):
    if not actual and not pred: return 1.0
    if not actual or not pred: return 0.0
    tp = len(pred & actual)
    fp = len(pred - actual)
    fn = len(actual - pred)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    if prec+rec == 0: return 0.0
    return 1.25*prec*rec/(0.25*prec+rec)

print(f"Loading {N_SAMPLE} S1 training entities...")
t0 = time.time()

s1_sample = {}
s1_order = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= N_SAMPLE: break
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4:
            sid, name, addr, country = p[0], p[1], p[2], p[3]
            s1_sample[sid] = (name, addr, country)
            s1_order.append(sid)

gt = {}
s1_set = set(s1_order)
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 2 and p[0] in s1_set:
            m = p[1].strip()
            gt[p[0]] = set(m.split(',')) if m else set()
for sid in s1_order:
    if sid not in gt: gt[sid] = set()

countries = set(v[2] for v in s1_sample.values())
s1_by_country = defaultdict(list)
for sid in s1_order:
    name, addr, country = s1_sample[sid]
    s1_by_country[country].append((sid, name, addr))

n_match_gt = sum(1 for v in gt.values() if v)
print(f"GT: {len(gt)} entities, {n_match_gt} with matches ({100*n_match_gt/len(gt):.1f}%)")
print(f"Countries: {sorted(countries)}")
print(f"Loaded in {time.time()-t0:.1f}s")

# Build indexes in 1 pass
print("\nBuilding indexes (1-pass S2+S3 read)...")
t0 = time.time()

core_idx    = {c: defaultdict(list) for c in countries}
clean_idx   = {c: defaultdict(list) for c in countries}
compact_idx = {c: defaultdict(list) for c in countries}
token2_idx  = {c: defaultdict(list) for c in countries}
addr_idx    = {c: defaultdict(list) for c in countries}
postal_idx  = {c: defaultdict(list) for c in countries}
bigram_idx  = {c: defaultdict(list) for c in countries}
cand_recs   = {c: {} for c in countries}

for path in [TRAIN_S2, TRAIN_S3]:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 4: continue
            cid, name, addr, country = p[0], p[1], p[2], p[3]
            if country not in countries: continue
            cn = clean_str(name)
            core = get_core(cn)
            ca = clean_str(addr)
            cmp = compact_str(core)
            cand_recs[country][cid] = (core, cn, ca, cmp)
            if core: core_idx[country][core].append(cid)
            if cn: clean_idx[country][cn].append(cid)
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
            n += 1
    print(f"  {os.path.basename(path)}: {n:,} records")

print(f"Indexes built in {time.time()-t0:.1f}s")
for c in sorted(countries):
    print(f"  {c}: {len(cand_recs[c]):,} candidates, core_keys={len(core_idx[c]):,}")

# Run matching
print("\nMatching...")
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

all_predictions = {}
total_cands = 0

for country, s1_items in s1_by_country.items():
    ci=core_idx[country]; cli=clean_idx[country]; cpi=compact_idx[country]
    t2i=token2_idx[country]; ai=addr_idx[country]; pi=postal_idx[country]
    bi=bigram_idx[country]; cr=cand_recs[country]

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
        if core in ci: cands.update(ci[core][:50])
        if cn in cli: cands.update(cli[cn][:50])
        if cmp and cmp in cpi: cands.update(cpi[cmp][:30])
        if len(tokens) >= 2:
            k=(tokens[0],tokens[1])
            if k in t2i: cands.update(t2i[k][:30])
            bg=tokens[0][:4]+tokens[1][:4]
            if bg in bi: cands.update(bi[bg][:30])
        elif len(tokens)==1 and len(tokens[0])>=5:
            k=(tokens[0],"")
            if k in t2i: cands.update(t2i[k][:30])
        if ca:
            num,alpha=extract_addr_anchor(ca)
            if num and alpha and (num,alpha) in ai: cands.update(ai[(num,alpha)][:20])
        if postal and postal in pi: cands.update(pi[postal][:50])

        matched = set()
        for cid in cands:
            c_core,c_clean,cand_addr,c_cmp = cr.get(cid,("","","",""))
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
                if JaroWinkler.normalized_similarity(core,c_core)>=0.92: matched.add(cid); continue
            if postal and cand_addr:
                cp=extract_postal(cand_addr,country)
                if postal==cp and n_sim>=55: matched.add(cid); continue

        all_predictions[sid] = matched
        total_cands += len(cands)

# Compute F0.5
scores=[]
nc=nw=np_=npa=nm=0
for sid in s1_order:
    pred=all_predictions.get(sid,set())
    actual=gt.get(sid,set())
    sc=f05(pred,actual)
    scores.append(sc)
    if not actual:
        if not pred: nc+=1
        else: nw+=1
    else:
        if sc==1.0: np_+=1
        elif sc>0: npa+=1
        else: nm+=1

macro=sum(scores)/len(scores)
ac=total_cands/len(s1_order)
print(f"\n{'='*60}")
print(f"  LOCAL EVAL (v3 pipeline logic, N={N_SAMPLE})")
print(f"{'='*60}")
print(f"  Macro F0.5 Score:      {macro:.4f}")
print(f"  Avg candidates/entity: {ac:.1f}")
print(f"  Singletons correct:    {nc}")
print(f"  Singletons WRONG FP:   {nw}  <- false merges on singletons")
print(f"  Matched perfect (1.0): {np_}")
print(f"  Matched partial:       {npa}")
print(f"  Matched missed (0.0):  {nm}  <- recall failures")
print(f"{'='*60}")
