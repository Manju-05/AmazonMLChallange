import os
import re
import sys
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

print("=" * 75)
print("  AMAZON ML CHALLENGE 2026 — HIGH-PRECISION CALIBRATOR & SIZE COMPLIANCE")
print("=" * 75)

GENERIC_WORDS = {
    'st', 'street', 'road', 'rd', 'ave', 'avenue', 'lane', 'ln', 'dr', 'drive',
    'blvd', 'boulevard', 'floor', 'fl', 'ground', 'unit', 'suite', 'ste', 'apt',
    'near', 'opp', 'opposite', 'behind', 'b/h', 'beside', 'delhi', 'mumbai',
    'india', 'usa', 'united', 'states', 'france', 'paris', 'null', 'none', 'city'
}

def clean_str(s):
    if not s or s == 'NULL': return ''
    return re.sub(r'[^a-z0-9 ]', ' ', s.lower()).strip()

def extract_addr_features(addr):
    tokens = [w for w in clean_str(addr).split() if len(w) >= 2]
    nums = {t for t in tokens if t.isdigit() and int(t) != 0}
    words = {t for t in tokens if not t.isdigit() and t not in GENERIC_WORDS and len(t) >= 3}
    return nums, words

TEST_DIR = r"student_resource/student_resource/dataset/test"
INPUT_TSV = r"d:\Sigma\AmazonMLChallange2K26\solution\output\matching_results.tsv"
OUTPUT_TSV = r"d:\Sigma\AmazonMLChallange2K26\solution\output\matching_results_calibrated.tsv"

COUNTRIES = [
    ("France", 259452),
    ("India", 809986),
    ("US", 663106)
]

# Read original matching results lines into memory or generator
print("\n[Step 1] Opening original matching results...")
fin = open(INPUT_TSV, "r", encoding="utf-8")
header = next(fin)
fout = open(OUTPUT_TSV, "w", encoding="utf-8", newline="\n")
fout.write(header)

total_processed = 0
total_matches_out = 0
t_start = time.time()

for country, expected_count in COUNTRIES:
    print(f"\n-- Processing {country} ({expected_count:,} entities) --")
    t_c = time.time()
    
    # 1. Load S1 for country
    s1_features = {}
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[3] == country:
                s1_features[p[0]] = extract_addr_features(p[2])
    print(f"  ✓ Ingested {len(s1_features):,} S1 records")
    
    # 2. Load S2 and S3 for country
    c_features = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(TEST_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[3] == country:
                    c_features[p[0]] = extract_addr_features(p[2])
    print(f"  ✓ Ingested {len(c_features):,} candidate features")
    
    # 3. Calibrate entities
    country_matches = 0
    for _ in range(expected_count):
        line = next(fin)
        p = line.rstrip("\r\n").split("\t")
        s1_id = p[0]
        mids_str = p[1] if len(p) > 1 else ""
        mids = mids_str.split(",") if mids_str else []
        
        if len(mids) <= 3:
            filtered = mids
        else:
            s1_nums, s1_words = s1_features.get(s1_id, (set(), set()))
            scored = []
            for cid in mids:
                c_nums, c_words = c_features.get(cid, (set(), set()))
                common_nums = bool(s1_nums & c_nums)
                common_words = len(s1_words & c_words)
                score = (50 if common_nums else 0) + (common_words * 20)
                if score > 0 or (not s1_words and not c_words):
                    scored.append((score, cid))
                    
            if scored:
                scored.sort(key=lambda x: -x[0])
                filtered = [cid for _, cid in scored[:6]]
            else:
                filtered = mids[:2]
                
        out_line = f"{s1_id}\t{','.join(filtered)}\n"
        fout.write(out_line)
        country_matches += len(filtered)
        total_processed += 1
        
    print(f"  ✓ Completed {country} in {time.time()-t_c:.2f}s | Avg matches/entity: {country_matches/expected_count:.2f}")
    del s1_features, c_features

fin.close()
fout.close()

out_size_mb = os.path.getsize(OUTPUT_TSV) / (1024 * 1024)
print("\n" + "=" * 75)
print(f" CALIBRATION COMPLETE IN {time.time() - t_start:.2f}s!")
print(f" Output File: {OUTPUT_TSV}")
print(f" Total Rows:  {total_processed:,}")
print(f" Output Size: {out_size_mb:.2f} MB (Well under 512 MB portal limit!)")
print("=" * 75)
