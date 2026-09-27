"""
pipeline.py — End-to-end Entity Resolution pipeline.

Orchestrates all phases:
1. Data loading
2. Preprocessing
3. Validation split
4. Blocking (candidate generation)
5. Feature engineering
6. Model training & threshold optimisation
7. Evaluation on validation set
8. Test inference & output generation
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import numpy as np
import pandas as pd
from typing import Dict, Set

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MATCHING_OUTPUT, CANDIDATE_OUTPUT, OUTPUT_DIR,
    VALIDATION_SPLIT, RANDOM_SEED
)
from data_loader import (
    load_training_data, load_test_data,
    create_validation_split, build_gt_dict
)
from preprocessor import preprocess_dataframe
from blocker import block_by_country
from feature_engine import compute_features_for_candidates, FEATURE_NAMES
from matcher import EntityMatcher
from evaluator import (
    evaluate_predictions, evaluate_blocking_recall,
    print_evaluation_report
)


def records_to_dict(df: pd.DataFrame) -> Dict[str, dict]:
    """Convert DataFrame to {entity_id: record_dict} for fast lookup."""
    return df.set_index("entity_id").to_dict("index")


def write_output_tsv(
    predictions: Dict[str, set],
    all_s1_ids: list,
    output_path: str,
    id_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
):
    """Write predictions to a tab-separated output file."""
    print(f"  Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{id_col}\t{match_col}\n")
        for s1_id in all_s1_ids:
            matched = predictions.get(s1_id, set())
            matched_str = ",".join(sorted(matched)) if matched else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    print(f"  ✓ Written {len(all_s1_ids):,} rows")


def run_training_pipeline():
    """
    Run the full training pipeline:
    Load → Preprocess → Split → Block → Features → Train → Evaluate
    """
    total_start = time.time()
    
    # ═══════════════════════════════════════════════
    # PHASE 1: Data Loading
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 1: DATA LOADING")
    print("="*70)
    
    s1, s2, s3, gt = load_training_data()
    
    # ═══════════════════════════════════════════════
    # PHASE 2: Preprocessing
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 2: PREPROCESSING")
    print("="*70)
    
    s1 = preprocess_dataframe(s1, "S1 records")
    s2 = preprocess_dataframe(s2, "S2 records")
    s3 = preprocess_dataframe(s3, "S3 records")
    
    # ═══════════════════════════════════════════════
    # PHASE 3: Validation Split
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 3: VALIDATION SPLIT")
    print("="*70)
    
    train_s1, val_s1, train_gt, val_gt = create_validation_split(s1, gt)
    
    train_gt_dict = build_gt_dict(train_gt)
    val_gt_dict   = build_gt_dict(val_gt)
    
    del s1, gt  # free memory
    gc.collect()
    
    # ═══════════════════════════════════════════════
    # PHASE 4: Blocking (Candidate Generation)
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 4: BLOCKING (CANDIDATE GENERATION)")
    print("="*70)
    
    # Block on training split first for feature generation
    print("\n[Training set blocking]")
    train_candidates = block_by_country(train_s1, s2, s3)
    
    # Evaluate blocking recall on training set
    train_recall, train_block_details = evaluate_blocking_recall(train_candidates, train_gt_dict)
    print(f"\n  Training blocking recall: {train_recall:.4f}")
    print(f"  Missed pairs: {train_block_details['missed_pairs']:,} / {train_block_details['total_true_pairs']:,}")
    
    # Block on validation split
    print("\n[Validation set blocking]")
    val_candidates = block_by_country(val_s1, s2, s3)
    
    val_recall, val_block_details = evaluate_blocking_recall(val_candidates, val_gt_dict)
    print(f"\n  Validation blocking recall: {val_recall:.4f}")
    print(f"  Missed pairs: {val_block_details['missed_pairs']:,} / {val_block_details['total_true_pairs']:,}")
    
    # ═══════════════════════════════════════════════
    # PHASE 5: Feature Engineering
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 5: FEATURE ENGINEERING")
    print("="*70)
    
    # Build record lookups
    train_s1_records = records_to_dict(train_s1)
    val_s1_records   = records_to_dict(val_s1)
    
    # Combined S2+S3 lookup
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_records = records_to_dict(s23)
    del s2, s3, s23
    gc.collect()
    
    # Compute features for training pairs
    print("\n[Computing training features]")
    X_train_df, y_train, train_pairs = compute_features_for_candidates(
        train_s1_records, s23_records, train_candidates,
        ground_truth=train_gt_dict, desc="Training features"
    )
    
    # Compute features for validation pairs
    print("\n[Computing validation features]")
    X_val_df, y_val, val_pairs = compute_features_for_candidates(
        val_s1_records, s23_records, val_candidates,
        ground_truth=val_gt_dict, desc="Validation features"
    )
    
    X_train = X_train_df.values.astype(np.float32)
    X_val   = X_val_df.values.astype(np.float32)
    feature_names = list(X_train_df.columns)
    
    del X_train_df, X_val_df, train_candidates
    gc.collect()
    
    # ═══════════════════════════════════════════════
    # PHASE 6: Model Training
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 6: MODEL TRAINING")
    print("="*70)
    
    matcher = EntityMatcher()
    matcher.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    # ═══════════════════════════════════════════════
    # PHASE 7: Threshold Optimisation & Evaluation
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" PHASE 7: THRESHOLD OPTIMISATION & EVALUATION")
    print("="*70)
    
    val_probas = matcher.predict_proba(X_val)
    
    best_threshold = matcher.optimise_threshold(
        val_probas, val_pairs, val_gt_dict
    )
    
    # Final evaluation at optimal threshold
    val_predictions = matcher.predict(X_val, val_pairs, threshold=best_threshold)
    
    # Ensure all val S1 entities are in predictions
    for s1_id in val_gt_dict:
        if s1_id not in val_predictions:
            val_predictions[s1_id] = set()
    
    macro_f05, details = evaluate_predictions(val_predictions, val_gt_dict)
    print_evaluation_report(macro_f05, details)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.lgb")
    matcher.save(model_path)
    
    total_time = time.time() - total_start
    print(f"\n  Total training pipeline time: {total_time/60:.1f} minutes")
    
    return matcher, best_threshold, feature_names


def run_test_pipeline(matcher: EntityMatcher, threshold: float):
    """
    Run inference on test data and generate submission files.
    """
    total_start = time.time()
    
    # ═══════════════════════════════════════════════
    # Load & Preprocess Test Data
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(" TEST INFERENCE")
    print("="*70)
    
    test_s1, test_s2, test_s3 = load_test_data()
    
    test_s1 = preprocess_dataframe(test_s1, "Test S1 records")
    test_s2 = preprocess_dataframe(test_s2, "Test S2 records")
    test_s3 = preprocess_dataframe(test_s3, "Test S3 records")
    
    # ═══════════════════════════════════════════════
    # Blocking on Test Data
    # ═══════════════════════════════════════════════
    print("\n[Test set blocking]")
    test_candidates = block_by_country(test_s1, test_s2, test_s3)
    
    # ═══════════════════════════════════════════════
    # Feature Engineering on Test Data
    # ═══════════════════════════════════════════════
    test_s1_records = records_to_dict(test_s1)
    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
    test_s23_records = records_to_dict(test_s23)
    del test_s2, test_s3, test_s23
    gc.collect()
    
    print("\n[Computing test features]")
    X_test_df, _, test_pairs = compute_features_for_candidates(
        test_s1_records, test_s23_records, test_candidates,
        ground_truth=None, desc="Test features"
    )
    
    X_test = X_test_df.values.astype(np.float32)
    del X_test_df
    gc.collect()
    
    # ═══════════════════════════════════════════════
    # Predict
    # ═══════════════════════════════════════════════
    print("\n[Generating predictions]")
    test_predictions = matcher.predict(X_test, test_pairs, threshold=threshold)
    
    # Ensure all test S1 entities are in predictions
    all_test_s1_ids = list(test_s1["entity_id"].values)
    for s1_id in all_test_s1_ids:
        if s1_id not in test_predictions:
            test_predictions[s1_id] = set()
    
    # ═══════════════════════════════════════════════
    # Write Output Files
    # ═══════════════════════════════════════════════
    print("\n[Writing output files]")
    
    # matching_results.tsv
    write_output_tsv(
        test_predictions, all_test_s1_ids,
        MATCHING_OUTPUT,
        id_col="source1_entity_id",
        match_col="matched_entity_ids",
    )
    
    # candidate_pairs.tsv
    write_output_tsv(
        test_candidates, all_test_s1_ids,
        CANDIDATE_OUTPUT,
        id_col="source1_entity_id",
        match_col="candidate_entity_ids",
    )
    
    total_time = time.time() - total_start
    print(f"\n  Total test pipeline time: {total_time/60:.1f} minutes")
    
    # Summary
    matched_count = sum(1 for v in test_predictions.values() if len(v) > 0)
    singleton_count = sum(1 for v in test_predictions.values() if len(v) == 0)
    total_links = sum(len(v) for v in test_predictions.values())
    
    print(f"\n  ═══ SUBMISSION SUMMARY ═══")
    print(f"  Total S1 entities: {len(all_test_s1_ids):,}")
    print(f"  Entities with matches: {matched_count:,}")
    print(f"  Singletons: {singleton_count:,}")
    print(f"  Total match links: {total_links:,}")
    print(f"  Output files:")
    print(f"    {MATCHING_OUTPUT}")
    print(f"    {CANDIDATE_OUTPUT}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Amazon ML Challenge 2026 — Business Entity Resolution     ║")
    print("║  End-to-End Pipeline                                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Train
    matcher, threshold, feature_names = run_training_pipeline()
    
    # Test inference
    run_test_pipeline(matcher, threshold)
    
    print("\n✅ Pipeline complete! Validate with:")
    print(f"   python utils/validate_submission.py --matching {MATCHING_OUTPUT} --candidate {CANDIDATE_OUTPUT} --test-dir dataset/test")
