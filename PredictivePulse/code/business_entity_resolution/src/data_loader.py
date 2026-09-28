"""
data_loader.py — Load and validate all TSV datasets.
Handles the tab-separated format and provides clean DataFrames.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import os
import gc
from tqdm import tqdm

from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    VALIDATION_SPLIT, RANDOM_SEED
)


def load_source(path: str, name: str = "") -> pd.DataFrame:
    """Load a source TSV file with proper tab separation."""
    print(f"  Loading {name or os.path.basename(path)}...", end=" ", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    # Fill NaN business_name / business_address with empty string
    df["business_name"]    = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"]          = df["country"].fillna("")
    print(f"{len(df):,} records")
    return df


def load_ground_truth(path: str = TRAIN_GT) -> pd.DataFrame:
    """Load ground truth mapping: source1_entity_id → matched_entity_ids."""
    print(f"  Loading ground truth...", end=" ", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    print(f"{len(df):,} entries")
    return df


def build_gt_dict(gt_df: pd.DataFrame) -> Dict[str, set]:
    """Convert ground truth DataFrame to dict: {s1_id: set(matched_ids)}."""
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if matched and matched.strip():
            gt_dict[s1_id] = set(matched.split(","))
        else:
            gt_dict[s1_id] = set()
    return gt_dict


def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training sources and ground truth."""
    print("[Data Loader] Loading training data...")
    s1 = load_source(TRAIN_S1, "Train Source 1")
    s2 = load_source(TRAIN_S2, "Train Source 2")
    s3 = load_source(TRAIN_S3, "Train Source 3")
    gt = load_ground_truth(TRAIN_GT)
    return s1, s2, s3, gt


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test sources."""
    print("[Data Loader] Loading test data...")
    s1 = load_source(TEST_S1, "Test Source 1")
    s2 = load_source(TEST_S2, "Test Source 2")
    s3 = load_source(TEST_S3, "Test Source 3")
    return s1, s2, s3


def create_validation_split(
    s1: pd.DataFrame,
    gt: pd.DataFrame,
    val_frac: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split S1 entities into train/val at the entity level.
    Returns: (train_s1, val_s1, train_gt, val_gt)
    """
    print(f"[Data Loader] Creating {val_frac:.0%} validation split (entity-level)...")
    
    s1_ids = s1["entity_id"].values
    np.random.seed(seed)
    np.random.shuffle(s1_ids)
    
    split_idx = int(len(s1_ids) * (1 - val_frac))
    train_ids = set(s1_ids[:split_idx])
    val_ids   = set(s1_ids[split_idx:])
    
    train_s1 = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1   = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    
    train_gt = gt[gt["source1_entity_id"].isin(train_ids)].reset_index(drop=True)
    val_gt   = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    
    print(f"  Train: {len(train_s1):,} S1 entities")
    print(f"  Val:   {len(val_s1):,} S1 entities")
    
    return train_s1, val_s1, train_gt, val_gt


def build_entity_lookup(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> Dict[str, dict]:
    """Build a combined lookup: entity_id → record dict, for fast access."""
    lookup = {}
    for df in [s1, s2, s3]:
        for _, row in df.iterrows():
            lookup[row["entity_id"]] = {
                "business_name": row["business_name"],
                "business_address": row["business_address"],
                "country": row["country"],
            }
    return lookup
