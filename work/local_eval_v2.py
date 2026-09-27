"""
local_eval_v2.py - Quick F0.5 evaluation of v2 pipeline logic on training data sample.
Tests 5,000 S1 entities from training set with ground truth.
"""
import sys, os, re, unicodedata, gc, time
from collections import defaultdict

sys.path.insert(0, os.path.join('solution', 'src'))

BASE = os.path.abspath('.')
TRAIN_S1 = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_source1.tsv')
TRAIN_S2 = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_source2.tsv')
TRAIN_S3 = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_source3.tsv')
TRAIN_GT = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'train', 'train_ground_truth.tsv')

N_SAMPLE = 5000  # number of S1 entities to evaluate

LEGAL_SUFFIXES = {
    "private", "limited", "llc", "incorporated", "corporation",
    "company", "pllc", "llp", "plc", "pvt", "ltd", "inc", "corp",
    "co", "sarl", "sas", "sci", "sa", "eurl", "snc", "groupe",
    "societe", "gmbh", "ag", "bv", "nv", "spa", "srl", "sl", "ms",
    "and", "associates", "enterprise", "enterprises", "services",
    "trading", "solutions", "technologies", "tech", "group",
    "international", "india", "national", "global",
}
STOPWORDS = {"the", "and", "of", "in", "for", "de", "du", "et", "la", "le", "des", "les", "en", "d", "l", "au", "aux", "a"}

RE_POSTAL_INDIA = re.compile(r'\b([1-9][0-9]{5})\b')
RE_POSTAL_US = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
RE_POSTAL_FRANCE = re.compile(r'\b([0-9]{5})\b')
RE_NUMBERS = re.compile(r'\b\d+\b')

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

def extract_addr_anchor(addr):
    nums = RE_NUMBERS.findall(addr)
    num = nums[0] if nums else ""
    toks = [t for t in addr.split() if t.isalpha() and len(t) >= 4]
    return num, (toks[0] if toks else "")

def extract_postal(addr, country):
    if not addr: return ""
    if country == "India": m = RE_POSTAL_INDIA.search(addr)
    elif country == "US": m = RE_POSTAL_US.search(addr)
    elif country == "France": m = RE_POSTAL_FRANCE.search(addr)
    else: m = RE_POSTAL_INDIA.search(addr) or RE_POSTAL_US.search(addr)
    return m.group(1) if m else ""

def f05(pred, actual):
    if not actual and not pred: return 1.0
    if not actual: return 0.0
    if not pred: return 0.0
    tp = len(pred & actual)
    fp = len(pred - actual)
    fn = len(actual - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0: return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)

print(f"Loading {N_SAMPLE} S1 training entities...")
t0 = time.time()

# Load a sample of S1
s1_sample = {}
s1_order = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= N_SAMPLE: break
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4:
            sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            s1_sample[sid] = (name, addr, country)
            s1_order.append(sid)

# Load GT for those S1 entities
gt = {}
s1_set = set(s1_order)
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 2 and parts[0] in s1_set:
            m = parts[1].strip()
            gt[parts[0]] = set(m.split(',')) if m else set()
for sid in s1_order:
    if sid not in gt: gt[sid] = set()

n_matched_gt = sum(1 for v in gt.values() if v)
print(f"GT: {len(gt)} entities, {n_matched_gt} with matches ({100*n_matched_gt/len(gt):.1f}%)")

# Get all S2/S3 IDs referenced in GT
all_gt_ids = set()
for v in gt.values():
    all_gt_ids.update(v)
gt_s2_ids = {x for x in all_gt_ids if x.startswith('S2-')}
gt_s3_ids = {x for x in all_gt_ids if x.startswith('S3-')}
print(f"GT references {len(all_gt_ids)} unique S2/S3 IDs")

# Group by country
s1_by_country = defaultdict(list)
for sid in s1_order:
    name, addr, country = s1_sample[sid]
    s1_by_country[country].append((sid, name, addr, country))

print(f"Country split: {dict((c, len(v)) for c,v in s1_by_country.items())}")
print(f"Data loaded in {time.time()-t0:.1f}s")

# Run v2 pipeline logic on training data
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

all_predictions = {}
total_candidates = 0

for country, s1_items in s1_by_country.items():
    print(f"\nProcessing {country}: {len(s1_items)} S1 entities...")
    t_c = time.time()
    
    # Build indexes from S2 and S3
    core_index = defaultdict(list)
    clean_index = defaultdict(list)
    compact_index = defaultdict(list)
    token2_index = defaultdict(list)
    addr_index = defaultdict(list)
    postal_index = defaultdict(list)
    bigram_index = defaultdict(list)
    cand_records = {}
    
    c_count = 0
    for path in [TRAIN_S2, TRAIN_S3]:
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 4 and parts[3] == country:
                    cid, name, addr = parts[0], parts[1], parts[2]
                    cn = clean_str(name)
                    core = get_core(cn)
                    ca = clean_str(addr)
                    cmp = compact_str(core)
                    cand_records[cid] = (core, cn, ca, cmp)
                    if core: core_index[core].append(cid)
                    if cn: clean_index[cn].append(cid)
                    if len(cmp) >= 6: compact_index[cmp].append(cid)
                    tokens = core.split()
                    if len(tokens) >= 2:
                        token2_index[(tokens[0], tokens[1])].append(cid)
                        bigram_index[tokens[0][:4]+tokens[1][:4]].append(cid)
                    elif len(tokens) == 1 and len(tokens[0]) >= 5:
                        token2_index[(tokens[0], "")].append(cid)
                    num, alpha = extract_addr_anchor(ca)
                    if num and alpha: addr_index[(num, alpha)].append(cid)
                    postal = extract_postal(ca, country)
                    if postal: postal_index[postal].append(cid)
                    c_count += 1
    
    print(f"  Indexed {c_count:,} candidates")
    
    for sid, name, addr, country2 in s1_items:
        cn = clean_str(name)
        core = get_core(cn)
        ca = clean_str(addr)
        cmp = compact_str(core)
        tokens = core.split()
        addr_toks = frozenset(ca.split())
        fuzzy_thresh = 85 if len(core) > 15 else 88
        postal = extract_postal(ca, country)
        
        cands = set()
        if core in core_index: cands.update(core_index[core][:50])
        if cn in clean_index: cands.update(clean_index[cn][:50])
        if cmp and cmp in compact_index: cands.update(compact_index[cmp][:30])
        if len(tokens) >= 2:
            if (tokens[0], tokens[1]) in token2_index:
                cands.update(token2_index[(tokens[0], tokens[1])][:30])
            bg = tokens[0][:4]+tokens[1][:4]
            if bg in bigram_index: cands.update(bigram_index[bg][:30])
        elif len(tokens) == 1 and len(tokens[0]) >= 5:
            if (tokens[0], "") in token2_index:
                cands.update(token2_index[(tokens[0], "")][:30])
        if ca:
            num, alpha = extract_addr_anchor(ca)
            if num and alpha and (num, alpha) in addr_index:
                cands.update(addr_index[(num, alpha)][:20])
        if postal and postal in postal_index:
            cands.update(postal_index[postal][:50])
        
        matched = set()
        for cid in cands:
            c_core, c_clean, cand_addr, c_cmp = cand_records.get(cid, ("","","",""))
            if core and core == c_core: matched.add(cid); continue
            if cn and cn == c_clean: matched.add(cid); continue
            if cmp and c_cmp and len(cmp)>=6 and len(c_cmp)>=6:
                if cmp in c_cmp or c_cmp in cmp: matched.add(cid); continue
            n_sim = fuzz.token_sort_ratio(core, c_core)
            if n_sim >= fuzzy_thresh: matched.add(cid); continue
            if ca and cand_addr:
                c_addr_toks = frozenset(cand_addr.split())
                ov = len(addr_toks & c_addr_toks)
                if ov >= 3 and n_sim >= 60: matched.add(cid); continue
                if ov >= 4: matched.add(cid); continue
            if core and c_core:
                jw = JaroWinkler.normalized_similarity(core, c_core)
                if jw >= 0.92: matched.add(cid); continue
            if postal and cand_addr:
                cp = extract_postal(cand_addr, country)
                if postal == cp and n_sim >= 55: matched.add(cid); continue
        
        all_predictions[sid] = matched
        total_candidates += len(cands)
    
    print(f"  Done in {time.time()-t_c:.1f}s")
    del core_index, clean_index, compact_index, token2_index, addr_index, postal_index, bigram_index, cand_records
    gc.collect()

# Compute F0.5
scores = []
n_correct_singletons = 0
n_wrong_singletons = 0
n_perfect = 0
n_partial = 0
n_missed = 0

for sid in s1_order:
    pred = all_predictions.get(sid, set())
    actual = gt.get(sid, set())
    score = f05(pred, actual)
    scores.append(score)
    if not actual:
        if not pred: n_correct_singletons += 1
        else: n_wrong_singletons += 1
    else:
        if score == 1.0: n_perfect += 1
        elif score > 0: n_partial += 1
        else: n_missed += 1

macro_f05 = sum(scores) / len(scores)
avg_cands = total_candidates / len(s1_order) if s1_order else 0

print(f"\n{'='*60}")
print(f"  LOCAL EVALUATION REPORT (v2 pipeline, N={N_SAMPLE})")
print(f"{'='*60}")
print(f"  Macro F0.5 Score:      {macro_f05:.4f}")
print(f"  Avg candidates/entity: {avg_cands:.1f}")
print(f"  Singletons correct:    {n_correct_singletons}")
print(f"  Singletons WRONG FP:   {n_wrong_singletons}")
print(f"  Matched perfect (1.0): {n_perfect}")
print(f"  Matched partial:       {n_partial}")
print(f"  Matched missed (0.0):  {n_missed}")
print(f"{'='*60}")
