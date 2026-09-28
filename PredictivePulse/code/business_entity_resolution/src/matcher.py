"""
matcher.py — ML classifier for entity matching.

Takes features from feature_engine.py and trains a LightGBM classifier
to predict whether a candidate pair is a true match.
Includes threshold optimisation for F₀.₅.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Set, Tuple, List, Optional
from tqdm import tqdm

from config import F_BETA, THRESHOLD_SWEEP, RANDOM_SEED
from evaluator import f_beta_score


class EntityMatcher:
    """LightGBM-based entity matcher with F₀.₅ threshold optimisation."""
    
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.feature_names = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
    ):
        """
        Train the LightGBM classifier.
        
        Uses class weights to handle severe label imbalance
        (most candidate pairs are non-matches).
        """
        self.feature_names = feature_names
        
        # Compute class weight: upweight positives
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        print(f"[Matcher] Training LightGBM classifier...")
        print(f"  Training samples: {len(y_train):,} ({n_pos:,} positive, {n_neg:,} negative)")
        print(f"  Scale pos weight: {scale_pos_weight:.1f}")
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_pos_weight,
            "min_child_samples": 50,
            "max_depth": -1,
            "random_state": RANDOM_SEED,
            "verbose": -1,
            "n_jobs": -1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        
        callbacks = [lgb.log_evaluation(period=100)]
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        print(f"  ✓ Training complete ({self.model.num_trees()} trees)")
        
        # Feature importance
        if feature_names:
            importance = self.model.feature_importance(importance_type="gain")
            feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
            print(f"\n  Top 10 features by gain:")
            for name, imp in feat_imp[:10]:
                print(f"    {name:30s} {imp:10.1f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability for each pair."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.predict(X)
    
    def optimise_threshold(
        self,
        probas: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        ground_truth: Dict[str, set],
        thresholds: List[float] = None,
    ) -> float:
        """
        Find the threshold that maximises F₀.₅ on validation data.
        
        Args:
            probas: predicted probabilities for each pair
            pair_ids: [(s1_id, cand_id), ...]
            ground_truth: {s1_id: set(true_match_ids)}
            thresholds: list of thresholds to try
        
        Returns:
            optimal_threshold
        """
        if thresholds is None:
            thresholds = THRESHOLD_SWEEP
        
        print(f"\n[Matcher] Optimising threshold (sweeping {len(thresholds)} values)...")
        
        # Pre-group candidate predictions by s1_id: s1_id -> list of (cand_id, prob)
        from collections import defaultdict
        grouped_preds = defaultdict(list)
        for (s1_id, cand_id), prob in zip(pair_ids, probas):
            grouped_preds[s1_id].append((cand_id, prob))
        
        best_threshold = 0.5
        best_f05 = 0.0
        
        for threshold in tqdm(thresholds, desc="  Threshold sweep"):
            scores = []
            for s1_id, actual in ground_truth.items():
                cands_with_probs = grouped_preds.get(s1_id, [])
                predicted = {cand_id for cand_id, prob in cands_with_probs if prob >= threshold}
                
                if len(actual) == 0 and len(predicted) == 0:
                    scores.append(1.0)
                elif len(actual) == 0 and len(predicted) > 0:
                    scores.append(0.0)
                elif len(actual) > 0 and len(predicted) == 0:
                    scores.append(0.0)
                else:
                    tp = len(predicted & actual)
                    fp = len(predicted - actual)
                    fn = len(actual - predicted)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    scores.append(f_beta_score(prec, rec, beta=F_BETA))
            
            macro_f05 = float(np.mean(scores))
            
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"  ✓ Optimal threshold: {best_threshold:.2f} (F₀.₅ = {best_f05:.4f})")
        
        return best_threshold
    
    def predict(
        self,
        X: np.ndarray,
        pair_ids: List[Tuple[str, str]],
        threshold: float = None,
    ) -> Dict[str, set]:
        """
        Generate final predictions: {s1_id: set(matched_ids)}.
        
        Args:
            X: feature matrix
            pair_ids: [(s1_id, cand_id), ...]
            threshold: classification threshold (uses optimised if None)
        
        Returns:
            {s1_entity_id: set(matched_entity_ids)}
        """
        if threshold is None:
            threshold = self.threshold
        
        probas = self.predict_proba(X)
        
        predictions = {}
        for (s1_id, cand_id), prob in zip(pair_ids, probas):
            if s1_id not in predictions:
                predictions[s1_id] = set()
            if prob >= threshold:
                predictions[s1_id].add(cand_id)
        
        return predictions
    
    def save(self, path: str):
        """Save model to file."""
        if self.model:
            self.model.save_model(path)
            print(f"  Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        print(f"  Model loaded from {path}")
