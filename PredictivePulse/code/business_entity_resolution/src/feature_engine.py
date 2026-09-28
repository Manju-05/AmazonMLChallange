"""
feature_engine.py — Compute pairwise features for candidate pairs.

For each (S1 entity, S2/S3 candidate) pair, compute similarity features
across names, addresses, and metadata. These features feed into the
matching classifier.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from tqdm import tqdm

from preprocessor import (
    extract_name_tokens, extract_addr_tokens, extract_numeric_tokens, strip_accents
)


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity of two sets."""
    if not set_a and not set_b:
        return 1.0  # both empty → "same"
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _overlap_coefficient(set_a: set, set_b: set) -> float:
    """Overlap coefficient: |A∩B| / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return intersection / min_size if min_size > 0 else 0.0


def _length_ratio(s1: str, s2: str) -> float:
    """Ratio of shorter to longer string length."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    return min(len1, len2) / max(len1, len2)


def compute_pair_features(
    s1_record: dict,
    cand_record: dict,
) -> dict:
    """
    Compute all pairwise features for a single (S1, candidate) pair.
    
    Both records should have keys:
    - name_clean, name_core, addr_clean, country, postal_code
    """
    features = {}
    
    s1_name  = s1_record.get("name_clean", "")
    c_name   = cand_record.get("name_clean", "")
    s1_core  = s1_record.get("name_core", "")
    c_core   = cand_record.get("name_core", "")
    s1_addr  = s1_record.get("addr_clean", "")
    c_addr   = cand_record.get("addr_clean", "")
    s1_postal = s1_record.get("postal_code", "")
    c_postal  = cand_record.get("postal_code", "")
    
    # ── Name Features ──
    
    # 1. Full name token Jaccard
    s1_name_tokens = extract_name_tokens(s1_name)
    c_name_tokens  = extract_name_tokens(c_name)
    features["name_jaccard"] = _jaccard(s1_name_tokens, c_name_tokens)
    
    # 2. Normalised Levenshtein ratio (rapidfuzz: 0-100 scale)
    features["name_levenshtein"] = fuzz.ratio(s1_name, c_name) / 100.0
    
    # 3. Jaro-Winkler similarity
    features["name_jaro_winkler"] = JaroWinkler.normalized_similarity(s1_name, c_name)
    
    # 4. Partial ratio (best local alignment)
    features["name_partial_ratio"] = fuzz.partial_ratio(s1_name, c_name) / 100.0
    
    # 5. Token sort ratio (order-invariant)
    features["name_token_sort"] = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    
    # 6. Token set ratio (handles subset/superset)
    features["name_token_set"] = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    
    # 7. Core name exact match
    features["name_core_exact"] = 1.0 if (s1_core and c_core and s1_core == c_core) else 0.0
    
    # 8. Core name Jaccard
    s1_core_tokens = extract_name_tokens(s1_core)
    c_core_tokens  = extract_name_tokens(c_core)
    features["name_core_jaccard"] = _jaccard(s1_core_tokens, c_core_tokens)
    
    # 9. Core name Levenshtein
    features["name_core_levenshtein"] = fuzz.ratio(s1_core, c_core) / 100.0
    
    # 10. Name length ratio
    features["name_len_ratio"] = _length_ratio(s1_name, c_name)
    
    # 11. Name token overlap coefficient
    features["name_token_overlap"] = _overlap_coefficient(s1_name_tokens, c_name_tokens)
    
    # 12. First 3 chars match
    features["name_first3_match"] = 1.0 if (
        len(s1_core) >= 3 and len(c_core) >= 3 and s1_core[:3] == c_core[:3]
    ) else 0.0
    
    # 13. Accent-stripped name match (for French names)
    s1_stripped = strip_accents(s1_name)
    c_stripped  = strip_accents(c_name)
    features["name_accent_stripped_ratio"] = fuzz.ratio(s1_stripped, c_stripped) / 100.0
    
    # ── Address Features ──
    
    # 14. Address available for both?
    s1_has_addr = 1.0 if s1_addr else 0.0
    c_has_addr  = 1.0 if c_addr else 0.0
    features["addr_has_both"] = s1_has_addr * c_has_addr
    
    if s1_addr and c_addr:
        s1_addr_tokens = extract_addr_tokens(s1_addr)
        c_addr_tokens  = extract_addr_tokens(c_addr)
        
        # 15. Address token Jaccard
        features["addr_jaccard"] = _jaccard(s1_addr_tokens, c_addr_tokens)
        
        # 16. Address Levenshtein
        features["addr_levenshtein"] = fuzz.ratio(s1_addr, c_addr) / 100.0
        
        # 17. Address token sort ratio
        features["addr_token_sort"] = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        
        # 18. Address token set ratio
        features["addr_token_set"] = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
        
        # 19. Address token overlap
        features["addr_token_overlap"] = _overlap_coefficient(s1_addr_tokens, c_addr_tokens)
        
        # 20. Numeric token overlap (house numbers, etc.)
        s1_nums = extract_numeric_tokens(s1_addr)
        c_nums  = extract_numeric_tokens(c_addr)
        features["addr_number_overlap"] = _jaccard(s1_nums, c_nums)
        
        # 21. Address length ratio
        features["addr_len_ratio"] = _length_ratio(s1_addr, c_addr)
    else:
        # No address comparison possible
        features["addr_jaccard"] = 0.0
        features["addr_levenshtein"] = 0.0
        features["addr_token_sort"] = 0.0
        features["addr_token_set"] = 0.0
        features["addr_token_overlap"] = 0.0
        features["addr_number_overlap"] = 0.0
        features["addr_len_ratio"] = 0.0
    
    # 22. Postal code match
    features["postal_code_match"] = 1.0 if (
        s1_postal and c_postal and s1_postal == c_postal
    ) else 0.0
    
    # 23. Candidate source (S2=0, S3=1)
    cand_id = cand_record.get("entity_id", "")
    features["source_is_s3"] = 1.0 if cand_id.startswith("S3-") else 0.0
    
    # 24. Combined name + addr score (weighted average)
    name_score = features["name_token_sort"]
    addr_score = features["addr_token_sort"] if features["addr_has_both"] else 0.0
    if features["addr_has_both"]:
        features["combined_score"] = 0.6 * name_score + 0.4 * addr_score
    else:
        features["combined_score"] = name_score
    
    return features


def compute_features_for_candidates(
    s1_records: Dict[str, dict],
    candidate_records: Dict[str, dict],
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, set] = None,
    desc: str = "Computing features",
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
    """
    Compute features for all candidate pairs.
    
    Args:
        s1_records: {entity_id: record_dict} for S1 entities
        candidate_records: {entity_id: record_dict} for S2/S3 entities
        candidates: {s1_id: set(candidate_ids)}
        ground_truth: optional {s1_id: set(true_match_ids)} for labels
    
    Returns:
        (feature_df, labels_array, pair_ids_list)
        - feature_df: DataFrame with one row per pair
        - labels_array: 1 for true match, 0 for non-match (or None if no GT)
        - pair_ids_list: [(s1_id, cand_id), ...]
    """
    all_features = []
    all_labels = []
    all_pairs = []
    
    # Count total pairs for progress bar
    total_pairs = sum(len(cands) for cands in candidates.values())
    
    pbar = tqdm(total=total_pairs, desc=desc)
    
    for s1_id, cand_ids in candidates.items():
        s1_rec = s1_records.get(s1_id)
        if s1_rec is None:
            pbar.update(len(cand_ids))
            continue
        
        true_matches = ground_truth.get(s1_id, set()) if ground_truth else None
        
        for cand_id in cand_ids:
            cand_rec = candidate_records.get(cand_id)
            if cand_rec is None:
                pbar.update(1)
                continue
            
            feats = compute_pair_features(s1_rec, cand_rec)
            all_features.append(feats)
            all_pairs.append((s1_id, cand_id))
            
            if true_matches is not None:
                all_labels.append(1 if cand_id in true_matches else 0)
            
            pbar.update(1)
    
    pbar.close()
    
    feature_df = pd.DataFrame(all_features)
    labels = np.array(all_labels) if all_labels else None
    
    print(f"  ✓ Features computed for {len(all_features):,} pairs")
    if labels is not None:
        pos = labels.sum()
        neg = len(labels) - pos
        print(f"  Labels: {pos:,} positive, {neg:,} negative (ratio 1:{neg/max(pos,1):.0f})")
    
    return feature_df, labels, all_pairs


FEATURE_NAMES = [
    "name_jaccard", "name_levenshtein", "name_jaro_winkler",
    "name_partial_ratio", "name_token_sort", "name_token_set",
    "name_core_exact", "name_core_jaccard", "name_core_levenshtein",
    "name_len_ratio", "name_token_overlap", "name_first3_match",
    "name_accent_stripped_ratio",
    "addr_has_both", "addr_jaccard", "addr_levenshtein",
    "addr_token_sort", "addr_token_set", "addr_token_overlap",
    "addr_number_overlap", "addr_len_ratio",
    "postal_code_match", "source_is_s3", "combined_score",
]
