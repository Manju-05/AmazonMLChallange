import pandas as pd
import numpy as np
import time, re, unicodedata
from rapidfuzz import fuzz

t0 = time.time()
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

s1_df = pd.read_csv('student_resource/student_resource/dataset/test/test_source1.tsv', sep='\t', nrows=5000)

s1_cands = {}
all_needed_cids = set()
with open('solution/output/candidate_pairs.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 5000: break
        parts = line.strip().split('\t')
        sid = parts[0]
        cids = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        s1_cands[sid] = cids
        all_needed_cids.update(cids)

cand_recs = {}
for path in ['test_source2.tsv', 'test_source3.tsv']:
    with open('student_resource/student_resource/dataset/test/' + path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in all_needed_cids:
                cid, name, addr, country = p[0], p[1], p[2], p[3]
                cn = clean_name(name)
                ca = clean_name(addr)
                core = get_core(cn)
                post = get_postal(ca, country)
                num = get_anchor_num(ca)
                cand_recs[cid] = (core, cn, ca, post, num)

all_s1_scores = []
for _, row in s1_df.iterrows():
    sid = row['entity_id']
    country = str(row['country'])
    cn = clean_name(str(row['business_name']))
    ca = clean_name(str(row['business_address']))
    core = get_core(cn)
    toks = frozenset(ca.split())
    post = get_postal(ca, country)
    num = get_anchor_num(ca)
    
    cands = s1_cands.get(sid, [])
    scored = []
    
    for cid in cands:
        rec = cand_recs.get(cid)
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
    all_s1_scores.append(scored)

print("="*65)
print("THRESHOLD & TOP-K SWEEP (Ground Truth Target: Mean=3.46, Singletons=5.6%):")
print(f"{'Threshold':>10} | {'Max-K':>5} | {'Mean Matches':>12} | {'Singletons %':>12} | {'Median':>6}")
print("-"*65)

for th in [0.80, 0.83, 0.85, 0.88, 0.90]:
    for k in [3, 4, 5]:
        counts = []
        sing = 0
        for scored in all_s1_scores:
            c = [cid for sc, cid in scored[:k] if sc >= th]
            counts.append(len(c))
            if len(c) == 0: sing += 1
        cnts = np.array(counts)
        print(f"{th:10.2f} | {k:5d} | {np.mean(cnts):12.2f} | {100*sing/len(cnts):11.1f}% | {np.median(cnts):6.1f}")
print("="*65)
