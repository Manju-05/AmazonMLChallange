"""
smoke_test.py — Quick end-to-end pipeline test on a small subset.
Validates all modules work together before running the full pipeline.
"""
import sys
import io
import os
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from preprocessor import preprocess_dataframe
from blocker import block_by_country
from feature_engine import compute_features_for_candidates
from matcher import EntityMatcher
from evaluator import evaluate_predictions, evaluate_blocking_recall

# ── Load tiny subset ──────────────────────────────────────────────
N = 2000
print(f"Loading {N} rows from each source...")
s1 = pd.read_csv(TRAIN_S1, sep='\t', dtype=str, nrows=N)
s2 = pd.read_csv(TRAIN_S2, sep='\t', dtype=str, nrows=N * 3)
s3 = pd.read_csv(TRAIN_S3, sep='\t', dtype=str, nrows=N * 3)
gt_raw = pd.read_csv(TRAIN_GT, sep='\t', dtype=str)

for df in [s1, s2, s3]:
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
gt_raw['matched_entity_ids'] = gt_raw['matched_entity_ids'].fillna('')

# ── Preprocessing ─────────────────────────────────────────────────
print("\nPreprocessing...")
s1 = preprocess_dataframe(s1, 'S1')
s2 = preprocess_dataframe(s2, 'S2')
s3 = preprocess_dataframe(s3, 'S3')

# ── Build GT dict ─────────────────────────────────────────────────
s1_ids = set(s1['entity_id'].values)
gt_rows = gt_raw[gt_raw['source1_entity_id'].isin(s1_ids)]
gt_dict = {}
for _, row in gt_rows.iterrows():
    m = row['matched_entity_ids']
    gt_dict[row['source1_entity_id']] = set(m.split(',')) if m.strip() else set()
for eid in s1_ids:
    if eid not in gt_dict:
        gt_dict[eid] = set()

with_matches = sum(1 for v in gt_dict.values() if v)
print(f"GT entries: {len(gt_dict)}, with matches: {with_matches}")

# ── Blocking ──────────────────────────────────────────────────────
print("\nBlocking...")
candidates = block_by_country(s1, s2, s3, top_k=20, min_score=0.2)
recall, bd = evaluate_blocking_recall(candidates, gt_dict)
avg_cands = bd['avg_candidates_per_entity']
print(f"Blocking recall: {recall:.3f} | Avg candidates: {avg_cands:.1f}")

# ── Feature Engineering ───────────────────────────────────────────
print("\nFeature Engineering...")
s23 = pd.concat([s2, s3], ignore_index=True)
s1_recs = s1.set_index('entity_id').to_dict('index')
s23_recs = s23.set_index('entity_id').to_dict('index')

all_s1 = list(s1_ids)
np.random.seed(42)
np.random.shuffle(all_s1)
tr_ids = set(all_s1[:1600])
va_ids = set(all_s1[1600:])

tr_cands = {k: v for k, v in candidates.items() if k in tr_ids}
va_cands = {k: v for k, v in candidates.items() if k in va_ids}
tr_gt = {k: v for k, v in gt_dict.items() if k in tr_ids}
va_gt = {k: v for k, v in gt_dict.items() if k in va_ids}

X_tr_df, y_tr, tr_pairs = compute_features_for_candidates(
    s1_recs, s23_recs, tr_cands, tr_gt, 'Train')
X_va_df, y_va, va_pairs = compute_features_for_candidates(
    s1_recs, s23_recs, va_cands, va_gt, 'Val')

X_tr = X_tr_df.values.astype(np.float32)
X_va = X_va_df.values.astype(np.float32)

# ── Model Training ────────────────────────────────────────────────
print("\nTraining LightGBM...")
m = EntityMatcher()
m.train(X_tr, y_tr, X_va, y_va, feature_names=list(X_tr_df.columns))

probas = m.predict_proba(X_va)
best_t = m.optimise_threshold(probas, va_pairs, va_gt)
preds = m.predict(X_va, va_pairs, threshold=best_t)
for s1_id in va_gt:
    if s1_id not in preds:
        preds[s1_id] = set()

f05, det = evaluate_predictions(preds, va_gt)
print(f"\nValidation F0.5: {f05:.4f}")
print(f"  Singletons correct: {det['singletons_correct']}")
print(f"  Singletons wrong:   {det['singletons_wrong']}")
print(f"  Matched perfect:    {det['matched_perfect']}")
print(f"  Matched partial:    {det['matched_partial']}")
print(f"  Matched missed:     {det['matched_missed']}")
print("\n✅  SMOKE TEST PASSED")
