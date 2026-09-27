# Business Entity Resolution — Solution

## Overview
ML solution for the Amazon ML Challenge 2026 Business Entity Resolution challenge. Matches business records across 3 noisy sources using a multi-stage pipeline: preprocessing → blocking → feature engineering → LightGBM classification.

## Architecture

```
Pipeline Flow:
Raw TSV Data
    ↓
Preprocessing (name/address normalisation, suffix standardisation, accent stripping)
    ↓
Country-Partitioned Blocking (TF-IDF char n-grams + exact core name match)
    ↓
24-Feature Pairwise Similarity (rapidfuzz, Jaccard, Jaro-Winkler, token sets)
    ↓
LightGBM Classifier (scale_pos_weight for imbalance)
    ↓
F₀.₅ Threshold Optimisation (validation sweep)
    ↓
Output: matching_results.tsv + candidate_pairs.tsv
```

## Project Structure

```
solution/
├── src/
│   ├── config.py          # Paths, constants, hyperparameters
│   ├── data_loader.py     # TSV loading, validation split
│   ├── preprocessor.py    # Text normalisation pipeline
│   ├── blocker.py         # TF-IDF blocking + exact name blocking
│   ├── feature_engine.py  # 24 pairwise similarity features
│   ├── matcher.py         # LightGBM classifier + threshold tuning
│   ├── evaluator.py       # F₀.₅ scorer
│   └── pipeline.py        # End-to-end orchestration
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (train + test)
```bash
cd solution/src
python pipeline.py
```

### 3. Validate output
```bash
cd student_resource/student_resource
python utils/validate_submission.py --matching ../../solution/output/matching_results.tsv --candidate ../../solution/output/candidate_pairs.tsv --test-dir dataset/test
```

## Key Design Decisions

1. **Character n-gram TF-IDF blocking** — language-agnostic, handles French zero-shot
2. **Country partitioning** — eliminates cross-country false positives, reduces compute
3. **Multi-strategy blocking union** — combines TF-IDF + exact name for high recall ceiling
4. **RapidFuzz features** — C-optimised string similarity (10x faster than pure Python)
5. **F₀.₅ threshold sweep** — directly optimises the leaderboard metric
6. **LightGBM** — fast, interpretable, Apache 2.0 licensed, tiny model (~100KB)
