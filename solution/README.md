# Business Entity Resolution — Solution Architecture & Runbook

This directory contains the machine learning and algorithmic entity resolution system for the **Amazon ML Challenge 2026**.

---

## 📋 Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Execution Modes](#execution-modes)
3. [Module Details & Architecture](#module-details--architecture)
4. [Hyperparameter Configuration (`config.py`)](#hyperparameter-configuration-configpy)
5. [Step-by-Step Runbook for Team Members](#step-by-step-runbook-for-team-members)
6. [Submission Output Specifications](#submission-output-specifications)

---

## 🔍 Pipeline Overview

The solution operates as a multi-stage entity resolution engine designed to handle **11.7 million test records** across three heterogeneous sources ($S_1, S_2, S_3$) spanning **India, the United States, and France**:

```
Raw TSV Data
   │
   ▼
[1] Multilingual Normalization (NFKD Unicode, 30+ legal suffixes, EN/FR stopwords, address anchors)
   │
   ▼
[2] Country Partitioning (Strict national segregation: India, US, France)
   │
   ▼
[3] 5-View Inverted Hash Blocking (Exact clean, core stem, compact string, token-pairs, address anchors)
   │
   ▼
[4] Precision-Calibrated Resolution (RapidFuzz C++ token similarity, address overlap, exact stems)
   │
   ▼
[5] Serialized Submission Outputs (1,732,544 rows in matching_results.tsv and candidate_pairs.tsv)
```

---

## ⚡ Execution Modes

We provide two distinct execution pipelines depending on your objective:

### Mode A: Production Fast Pipeline (`src/pipeline_production.py`) — **Recommended for Submissions**
- **Objective**: Complete end-to-end resolution on all 1.73M test instances with strict submission compliance.
- **Runtime**: **~7.2 minutes (433s)** on standard multi-core CPU.
- **Peak RAM**: **< 5 GB** (uses streaming I/O and per-country garbage collection).
- **Run command**:
  ```bash
  python solution/src/pipeline_production.py
  ```
- **Outputs**: Generates `solution/output/matching_results.tsv` and `solution/output/candidate_pairs.tsv`.

### Mode B: Modular ML Training & Evaluation Pipeline (`src/pipeline.py`) — **For Experiments & Model Tuning**
- **Objective**: Trains a LightGBM classifier on sampled training pairs, tunes the decision threshold on a validation split to maximize Macro $F_{0.5}$, and generates predictions.
- **Run command**:
  ```bash
  python solution/src/pipeline.py
  ```

---

## 🛠️ Module Details & Architecture

All code resides in [`solution/src/`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/):

| File | Purpose | Key Classes & Functions |
| :--- | :--- | :--- |
| [`config.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/config.py) | Central parameters, paths, and weights. | `LEGAL_SUFFIXES`, `STOPWORDS`, `LGBM_PARAMS`, `THRESHOLD_GRID` |
| [`data_loader.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/data_loader.py) | Streamed ingestion of large TSV files and train/val split creation. | `load_tsv_in_chunks()`, `create_validation_split()`, `load_ground_truth()` |
| [`preprocessor.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/preprocessor.py) | Text normalization across multiple languages and jurisdictions. | `clean_text()`, `strip_legal_suffixes()`, `get_core_entity_name()`, `extract_address_tokens()` |
| [`blocker.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/blocker.py) | Candidate generation using sparse TF-IDF and inverted indices. | `TfidfBlocker`, `ExactNameBlocker`, `generate_candidate_pairs()` |
| [`feature_engine.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/feature_engine.py) | Computes 24 pairwise similarity features between records. | `extract_pair_features()`, `batch_feature_extraction()` |
| [`matcher.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/matcher.py) | LightGBM classifier training and Macro $F_{0.5}$ threshold sweep. | `EntityMatcher`, `fit()`, `optimize_threshold()`, `predict()` |
| [`evaluator.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/evaluator.py) | Official competition Macro $F_{0.5}$ scorer. | `compute_macro_f05()`, `evaluate_predictions()` |
| [`pipeline_production.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/pipeline_production.py) | High-speed linear-time resolution runner. | `run_pipeline()` |

---

## ⚙️ Hyperparameter Configuration (`config.py`)

Key parameters in `config.py` that can be adjusted for experiments:

```python
# Blocking parameters
TFIDF_MAX_FEATURES = 100_000   # Max vocabulary for character n-grams
TFIDF_NGRAM_RANGE = (2, 4)      # Sub-word character n-grams
BLOCKING_TOP_K = 30             # Number of nearest candidates per entity
BLOCKING_MIN_SIM = 0.25         # Cosine similarity cutoff for candidates

# Matching parameters (Targeting Macro F0.5)
DECISION_THRESHOLD = 0.58       # Probability threshold favoring precision
RAPIDFUZZ_FUZZY_THRESHOLD = 88  # Minimum token-sort ratio for fuzzy link
ADDRESS_OVERLAP_MIN = 3         # Minimum shared address words
```

---

## 📖 Step-by-Step Runbook for Team Members

### 1. Install Environment Dependencies
```bash
pip install -r solution/requirements.txt
```

### 2. Run the Submission Pipeline
```bash
python solution/src/pipeline_production.py
```
Watch the terminal for progress updates per country:
```
── Processing Country: France (259,000 S1 entities) ──
  ✓ Indexed candidates...
  ✓ Matched entities...
── Processing Country: India (810,317 S1 entities) ──
  ✓ Indexed candidates...
  ✓ Matched entities...
── Processing Country: United States (663,227 S1 entities) ──
  ✓ Indexed candidates...
  ✓ Matched entities...
[Phase 3] Serializing submission TSV files...
PIPELINE FINISHED SUCCESSFULLY!
```

### 3. Verify the Outputs
Run the official Amazon submission validator:
```bash
python student_resource/student_resource/utils/validate_submission.py \
    --matching solution/output/matching_results.tsv \
    --candidate solution/output/candidate_pairs.tsv \
    --test-dir student_resource/student_resource/dataset/test
```
Ensure you see:
```
Status: PASS — no blocking issues found. Safe to submit.
```

---

## 📦 Submission Output Specifications

The pipeline outputs two tab-separated UTF-8 files in `solution/output/`:

### 1. `matching_results.tsv`
- **Columns**: `source1_entity_id \t matched_entity_ids`
- **Rows**: Exactly 1,732,544 rows + 1 header row.
- **Value format**: Comma-separated list of matched entity IDs from $S_2$ and $S_3$ (e.g. `s2_102,s3_904`).
- **Singletons**: Empty string for entities with no predicted match.

### 2. `candidate_pairs.tsv`
- **Columns**: `source1_entity_id \t candidate_entity_ids`
- **Rows**: Exactly 1,732,544 rows + 1 header row.
- **Value format**: Comma-separated list of candidate entity IDs.
- **Integrity Rule**: `matched_entity_ids` must be a strict subset of `candidate_entity_ids` for every row.
