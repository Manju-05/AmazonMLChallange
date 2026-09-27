"""
evaluator.py — F₀.₅ scoring and evaluation utilities.

Implements the exact macro-averaged F₀.₅ metric used by the leaderboard:
- Compute per-entity F₀.₅ (including singletons)
- Macro-average across all S1 entities
"""
from typing import Dict, Set, List, Tuple
import numpy as np


def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)


def compute_entity_f05(predicted: set, actual: set) -> float:
    """
    Compute F₀.₅ for a single S1 entity.
    
    Special cases:
    - Singleton (actual empty) + predicted empty → 1.0
    - Singleton (actual empty) + predicted non-empty → 0.0
    - Non-singleton (actual non-empty) + predicted empty → 0.0
    """
    # Both empty: singleton correctly predicted
    if len(actual) == 0 and len(predicted) == 0:
        return 1.0
    
    # Singleton but predicted matches: false positive
    if len(actual) == 0 and len(predicted) > 0:
        return 0.0
    
    # Has matches but predicted none: all false negatives
    if len(actual) > 0 and len(predicted) == 0:
        return 0.0
    
    # Both non-empty: compute precision, recall, F_0.5
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return f_beta_score(precision, recall, beta=0.5)


def evaluate_predictions(
    predictions: Dict[str, set],
    ground_truth: Dict[str, set],
) -> Tuple[float, dict]:
    """
    Compute macro-averaged F₀.₅ across all S1 entities.
    
    Args:
        predictions: {s1_entity_id: set(matched_ids)}
        ground_truth: {s1_entity_id: set(matched_ids)}
    
    Returns:
        (macro_f05, details_dict)
    """
    scores = []
    n_singletons_correct = 0
    n_singletons_wrong = 0
    n_matched_correct = 0
    n_matched_partial = 0
    n_matched_missed = 0
    
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        score = compute_entity_f05(predicted, actual)
        scores.append(score)
        
        # Categorise for diagnostics
        if len(actual) == 0:
            if len(predicted) == 0:
                n_singletons_correct += 1
            else:
                n_singletons_wrong += 1
        else:
            if score == 1.0:
                n_matched_correct += 1
            elif score > 0:
                n_matched_partial += 1
            else:
                n_matched_missed += 1
    
    macro_f05 = np.mean(scores) if scores else 0.0
    
    details = {
        "macro_f05": macro_f05,
        "total_entities": len(scores),
        "mean_score": macro_f05,
        "median_score": float(np.median(scores)) if scores else 0.0,
        "singletons_correct": n_singletons_correct,
        "singletons_wrong": n_singletons_wrong,
        "matched_perfect": n_matched_correct,
        "matched_partial": n_matched_partial,
        "matched_missed": n_matched_missed,
    }
    
    return macro_f05, details


def evaluate_blocking_recall(
    candidates: Dict[str, set],
    ground_truth: Dict[str, set],
) -> Tuple[float, dict]:
    """
    Measure blocking quality: what fraction of true matches appear in candidates?
    
    This is the RECALL CEILING — the maximum recall any downstream model can achieve.
    """
    total_true_pairs = 0
    found_pairs = 0
    missed_pairs = 0
    entities_with_missing = 0
    
    for s1_id, actual in ground_truth.items():
        if len(actual) == 0:
            continue  # singletons don't matter for blocking recall
        
        cands = candidates.get(s1_id, set())
        found = len(actual & cands)
        missed_count = len(actual - cands)
        
        total_true_pairs += len(actual)
        found_pairs += found
        missed_pairs += missed_count
        
        if missed_count > 0:
            entities_with_missing += 1
    
    pair_recall = found_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    
    # Compute candidates per entity (reduction ratio)
    total_candidates = sum(len(v) for v in candidates.values())
    avg_candidates = total_candidates / len(candidates) if candidates else 0
    
    details = {
        "pair_recall": pair_recall,
        "found_pairs": found_pairs,
        "missed_pairs": missed_pairs,
        "total_true_pairs": total_true_pairs,
        "entities_with_missing": entities_with_missing,
        "total_candidates": total_candidates,
        "avg_candidates_per_entity": avg_candidates,
    }
    
    return pair_recall, details


def print_evaluation_report(macro_f05: float, details: dict):
    """Print a formatted evaluation report."""
    print("\n" + "="*60)
    print(f"  EVALUATION REPORT")
    print("="*60)
    print(f"  Macro F₀.₅ Score:     {macro_f05:.4f}")
    print(f"  Total entities:       {details['total_entities']:,}")
    print(f"  Median entity score:  {details['median_score']:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  Singletons correct:   {details['singletons_correct']:,}")
    print(f"  Singletons wrong:     {details['singletons_wrong']:,}")
    print(f"  Matched perfect (1.0):{details['matched_perfect']:,}")
    print(f"  Matched partial:      {details['matched_partial']:,}")
    print(f"  Matched missed (0.0): {details['matched_missed']:,}")
    print("="*60)
