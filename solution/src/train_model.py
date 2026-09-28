"""
train_model.py — Train high-precision LightGBM Entity Matcher.

Trains a LightGBM GBDT on training pairs and saves the model artifact
to solution/models/lgbm_model.txt for high-speed inference.
"""
import sys, os, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocessor import clean_business_name, extract_core_name, clean_business_address
from feature_engine import compute_pair_features, FEATURE_NAMES
from evaluator import compute_entity_f05

TRAIN_DIR = os.path.join(os.path.dirname(BASE_DIR), "student_resource", "student_resource", "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.txt")


def train_lgbm():
    print("=" * 65)
    print(" TRAINING HIGH-PRECISION LIGHTGBM MATCHER")
    print("=" * 65)

    gt_train = {}
    gt_val = {}
    
    print("\n[Step 1] Loading ground truth entities...")
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            p = line.strip("\n").split("\t")
            s1_id = p[0]
            matches = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            if i < 25000:
                gt_train[s1_id] = matches
            elif i < 30000:
                gt_val[s1_id] = matches
            else:
                break

    all_s1 = set(gt_train.keys()) | set(gt_val.keys())
    all_needed_cands = set().union(*gt_train.values()) | set().union(*gt_val.values())
    print(f"  ✓ {len(gt_train):,} train entities, {len(gt_val):,} validation entities, {len(all_needed_cands):,} match targets.")

    print("\n[Step 2] Ingesting Source 1 records...")
    s1_records = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip("\n").split("\t")
            if p[0] in all_s1:
                cname = clean_business_name(p[1])
                caddr = clean_business_address(p[2])
                s1_records[p[0]] = {
                    'entity_id': p[0],
                    'name_clean': cname,
                    'name_core': extract_core_name(cname),
                    'addr_clean': caddr,
                    'country': p[3],
                    'postal_code': ''
                }

    print("\n[Step 3] Ingesting candidate pool (S2 & S3)...")
    cand_records = {}
    def load_cands_file(fname, extra_limit=50000):
        count = 0
        with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.strip("\n").split("\t")
                cid = p[0]
                if cid in all_needed_cands or count < extra_limit:
                    cname = clean_business_name(p[1])
                    caddr = clean_business_address(p[2])
                    cand_records[cid] = {
                        'entity_id': cid,
                        'name_clean': cname,
                        'name_core': extract_core_name(cname),
                        'addr_clean': caddr,
                        'country': p[3],
                        'postal_code': ''
                    }
                    if cid not in all_needed_cands:
                        count += 1

    load_cands_file("train_source2.tsv", 50000)
    load_cands_file("train_source3.tsv", 50000)
    print(f"  ✓ Total candidates loaded: {len(cand_records):,}")

    # Build candidate indexes
    core_idx = defaultdict(list)
    addr_num_idx = defaultdict(list)
    addr_word_idx = defaultdict(list)

    for cid, rec in cand_records.items():
        country = rec['country']
        core = rec['name_core']
        addr = rec['addr_clean']
        nums = [str(int(n)) for n in re.findall(r'\b\d+\b', addr) if int(n) != 0 and len(n) <= 6]
        words = [w for w in addr.split() if len(w) >= 4 and w.isalpha()]
        
        if core: core_idx[(country, core)].append(cid)
        for n in nums:
            for w in words[:2]:
                addr_num_idx[(country, n, w)].append(cid)
        if len(words) >= 2:
            addr_word_idx[(country, words[0], words[1])].append(cid)

    def get_candidates(rec):
        country = rec['country']
        core = rec['name_core']
        addr = rec['addr_clean']
        nums = [str(int(n)) for n in re.findall(r'\b\d+\b', addr) if int(n) != 0 and len(n) <= 6]
        words = [w for w in addr.split() if len(w) >= 4 and w.isalpha()]
        
        cands = set()
        if (country, core) in core_idx:
            cands.update(core_idx[(country, core)][:30])
        for n in nums:
            for w in words[:2]:
                if (country, n, w) in addr_num_idx:
                    cands.update(addr_num_idx[(country, n, w)][:20])
        if len(words) >= 2:
            key = (country, words[0], words[1])
            if key in addr_word_idx:
                cands.update(addr_word_idx[key][:20])
        return cands

    print("\n[Step 4] Extracting 24-dimensional feature matrix...")
    X_train = []
    y_train = []

    for s1_id, true_set in gt_train.items():
        s1_rec = s1_records.get(s1_id)
        if not s1_rec: continue
        
        for cid in true_set:
            if cid in cand_records:
                feats = compute_pair_features(s1_rec, cand_records[cid])
                X_train.append([feats[f] for f in FEATURE_NAMES])
                y_train.append(1)
                
        blocked = get_candidates(s1_rec)
        neg_count = 0
        for cid in blocked:
            if cid not in true_set and cid in cand_records:
                feats = compute_pair_features(s1_rec, cand_records[cid])
                X_train.append([feats[f] for f in FEATURE_NAMES])
                y_train.append(0)
                neg_count += 1
                if neg_count >= 4:
                    break

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  ✓ Training pairs: {len(X_train):,} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})")

    print("\n[Step 5] Fitting LightGBM GBDT...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 45,
        'max_depth': 7,
        'scale_pos_weight': 2.0,
        'verbose': -1,
        'random_state': 42
    }
    gbm = lgb.train(params, dtrain, num_boost_round=150)

    # Save model
    gbm.save_model(MODEL_PATH)
    print(f"  ✓ Model saved successfully to {MODEL_PATH}")

    # Validate on holdout set
    print("\n[Step 6] Validating on 5,000 holdout entities...")
    val_pairs = []
    val_s1_indices = []

    for s1_id in gt_val.keys():
        s1_rec = s1_records.get(s1_id)
        if not s1_rec: continue
        blocked = get_candidates(s1_rec)
        all_c = blocked | gt_val[s1_id]
        for cid in all_c:
            if cid in cand_records:
                feats = compute_pair_features(s1_rec, cand_records[cid])
                val_pairs.append([feats[f] for f in FEATURE_NAMES])
                val_s1_indices.append((s1_id, cid))

    X_val = np.array(val_pairs)
    val_probs = gbm.predict(X_val)

    s1_cand_probs = defaultdict(list)
    for (s1_id, cid), prob in zip(val_s1_indices, val_probs):
        s1_cand_probs[s1_id].append((cid, prob))

    for thresh in [0.50, 0.60, 0.65, 0.70, 0.75]:
        total_f05 = 0.0
        for s1_id, true_set in gt_val.items():
            preds = {cid for cid, p in s1_cand_probs.get(s1_id, []) if p >= thresh}
            f05 = compute_entity_f05(preds, true_set)
            total_f05 += f05
        score = total_f05 / len(gt_val)
        print(f"  * Threshold = {thresh:.2f} --> Validation Macro F0.5 = {score:.4f}")

    print("\n" + "=" * 65)
    print(" MODEL TRAINING AND VALIDATION COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    train_lgbm()
