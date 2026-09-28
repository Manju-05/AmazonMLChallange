"""
blocker.py — Candidate generation / blocking strategies.

This is the MOST CRITICAL module. Blocking determines the recall ceiling —
no downstream model can recover true matches lost at this stage.

Strategy: Multi-strategy union blocking with country-first partitioning.
1. Country filter (mandatory): only compare within same country
2. TF-IDF character n-gram cosine similarity on combined name+address
3. TF-IDF on name only (catches address-missing records)
4. Exact core name match (fast, high precision)
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict
from tqdm import tqdm
import gc

from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    BLOCKING_TOP_K, BLOCKING_MIN_SCORE, BATCH_SIZE
)


def _sparse_cosine_top_k(
    query_matrix: csr_matrix,
    index_matrix: csr_matrix,
    top_k: int,
    min_score: float = 0.0,
    batch_size: int = 2000,
) -> List[List[Tuple[int, float]]]:
    """
    For each query row, find top-K most similar index rows by cosine similarity.
    Uses batched sparse dot products with direct CSR nonzero extraction for high speed and low memory.
    
    Returns list of list of (index_idx, score) tuples.
    """
    n_queries = query_matrix.shape[0]
    results = []
    
    for start in tqdm(range(0, n_queries, batch_size), desc="    TF-IDF blocking"):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        
        # Cosine similarity = dot product (both matrices are L2-normalised by TfidfVectorizer)
        scores = batch.dot(index_matrix.T)  # csr_matrix of shape: (batch_size, n_index)
        
        # Extract top-K from CSR representation directly
        for i in range(scores.shape[0]):
            r_start = scores.indptr[i]
            r_end = scores.indptr[i + 1]
            if r_end == r_start:
                results.append([])
                continue
            
            row_indices = scores.indices[r_start:r_end]
            row_data = scores.data[r_start:r_end]
            
            # Filter by min_score
            mask = row_data >= min_score
            if not np.any(mask):
                results.append([])
                continue
            
            valid_indices = row_indices[mask]
            valid_scores = row_data[mask]
            
            n_valid = len(valid_scores)
            if n_valid > top_k:
                top_k_part = np.argpartition(valid_scores, -top_k)[-top_k:]
                sorted_part = top_k_part[np.argsort(valid_scores[top_k_part])[::-1]]
                results.append([(int(valid_indices[idx]), float(valid_scores[idx])) for idx in sorted_part])
            else:
                sorted_order = np.argsort(valid_scores)[::-1]
                results.append([(int(valid_indices[idx]), float(valid_scores[idx])) for idx in sorted_order])
    
    return results


def block_by_country(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
    min_score: float = BLOCKING_MIN_SCORE,
) -> Dict[str, Set[str]]:
    """
    Main blocking function. For each S1 entity, find candidate S2/S3 matches.
    
    Strategy:
    1. Partition all sources by country
    2. Within each country: build TF-IDF on combined text of S2+S3
    3. Query each S1 entity to get top-K candidates
    4. Also add exact core-name matches as bonus candidates
    
    Returns:
        {s1_entity_id: set(candidate_s2_s3_ids)}
    """
    candidates = defaultdict(set)
    
    # Combine S2 + S3 into one pool per country
    s23 = pd.concat([s2_df, s3_df], ignore_index=True)
    
    countries = sorted(s1_df["country"].unique())
    print(f"[Blocker] Running blocking for {len(countries)} countries: {countries}")
    
    for country in countries:
        print(f"\n  ── Country: {country} ──")
        
        # Filter by country
        s1_country = s1_df[s1_df["country"] == country].reset_index(drop=True)
        s23_country = s23[s23["country"] == country].reset_index(drop=True)
        
        print(f"    S1 entities: {len(s1_country):,}")
        print(f"    S2+S3 entities: {len(s23_country):,}")
        
        if len(s1_country) == 0 or len(s23_country) == 0:
            print("    ⚠ Skipping — no records for this country")
            continue
        
        # ── Strategy 1: TF-IDF on combined text (name + address) ──
        print("    [Strategy 1] TF-IDF on combined text (char n-grams)...")
        
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            dtype=np.float32,
        )
        
        # Fit on S2+S3, transform both
        s23_texts = s23_country["combined_text"].fillna("").values
        s23_tfidf = tfidf.fit_transform(s23_texts)
        
        s1_texts = s1_country["combined_text"].fillna("").values
        s1_tfidf = tfidf.transform(s1_texts)
        
        # L2-normalise for cosine similarity
        from sklearn.preprocessing import normalize
        s1_tfidf = normalize(s1_tfidf, norm="l2")
        s23_tfidf = normalize(s23_tfidf, norm="l2")
        
        # Find top-K for each S1 entity
        results = _sparse_cosine_top_k(
            s1_tfidf, s23_tfidf,
            top_k=top_k,
            min_score=min_score,
            batch_size=2000,
        )
        
        s1_ids = s1_country["entity_id"].values
        s23_ids = s23_country["entity_id"].values
        
        for i, matches in enumerate(results):
            s1_id = s1_ids[i]
            for idx, score in matches:
                candidates[s1_id].add(s23_ids[idx])
        
        del s1_tfidf, s23_tfidf, tfidf
        gc.collect()
        
        # ── Strategy 2: TF-IDF on name only (catches missing-address records) ──
        print("    [Strategy 2] TF-IDF on name only...")
        
        tfidf_name = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES // 2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        
        s23_names = s23_country["name_clean"].fillna("").values
        s23_name_tfidf = tfidf_name.fit_transform(s23_names)
        
        s1_names = s1_country["name_clean"].fillna("").values
        s1_name_tfidf = tfidf_name.transform(s1_names)
        
        s1_name_tfidf = normalize(s1_name_tfidf, norm="l2")
        s23_name_tfidf = normalize(s23_name_tfidf, norm="l2")
        
        name_results = _sparse_cosine_top_k(
            s1_name_tfidf, s23_name_tfidf,
            top_k=top_k // 2,  # fewer candidates from name-only
            min_score=min_score + 0.1,  # slightly higher threshold
            batch_size=2000,
        )
        
        for i, matches in enumerate(name_results):
            s1_id = s1_ids[i]
            for idx, score in matches:
                candidates[s1_id].add(s23_ids[idx])
        
        del s1_name_tfidf, s23_name_tfidf, tfidf_name
        gc.collect()
        
        # ── Strategy 3: Exact core name match ──
        print("    [Strategy 3] Exact core name match...")
        
        # Build index: core_name → list of S2/S3 entity IDs
        core_name_index = defaultdict(list)
        s23_cores = s23_country["name_core"].values
        s23_eids = s23_country["entity_id"].values
        for core, eid in zip(s23_cores, s23_eids):
            if isinstance(core, str) and len(core) >= 3:
                core_name_index[core].append(eid)
        
        exact_matches = 0
        s1_cores = s1_country["name_core"].values
        s1_eids = s1_country["entity_id"].values
        for core, eid in zip(s1_cores, s1_eids):
            if isinstance(core, str) and core in core_name_index:
                for match_id in core_name_index[core][:50]:  # cap at 50 per name
                    candidates[eid].add(match_id)
                    exact_matches += 1
        
        print(f"    ✓ Exact core name matches added: {exact_matches:,}")
        
        del core_name_index
        gc.collect()
    
    # Ensure every S1 entity has an entry (even if empty)
    for eid in s1_df["entity_id"].values:
        if eid not in candidates:
            candidates[eid] = set()
    
    # Stats
    total_candidates = sum(len(v) for v in candidates.values())
    non_empty = sum(1 for v in candidates.values() if len(v) > 0)
    avg_cands = total_candidates / len(candidates) if candidates else 0
    
    print(f"\n[Blocker] Blocking complete:")
    print(f"  Total S1 entities: {len(candidates):,}")
    print(f"  S1 entities with candidates: {non_empty:,} ({100*non_empty/len(candidates):.1f}%)")
    print(f"  Total candidate pairs: {total_candidates:,}")
    print(f"  Avg candidates per entity: {avg_cands:.1f}")
    
    return dict(candidates)
