# Amazon ML Challenge 2026 — Multilingual Business Entity Resolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution** problem.

---

## 📌 Problem Overview
Business entity resolution is the task of linking noisy records across heterogeneous data sources referring to the same real-world entity. 

- **Primary Source ($S_1$)**: Contains target entities from multiple countries (e.g. US, India, France).
- **Secondary Sources ($S_2, S_3$)**: Contain potential duplicate and matching entity records.
- **Challenges**: Cross-lingual variations, legal suffix differences, abbreviations, address permutations, missing values, noise, and severe class imbalance.
- **Evaluation Metric**: Macro-averaged $F_{0.5}$ score across all $S_1$ entities (favoring precision over recall).

---

## 🏗️ Architecture & Pipeline

```
Raw Multi-source TSVs (S1, S2, S3)
                │
                ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Multi-lingual Preprocessing         │
│  - Address normalisation & expansion         │
│  - Legal suffix standardisation              │
│  - Noise & accent character stripping        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Country-Partitioned Blocking       │
│  - Fast sparse CSR matrix dot product        │
│  - Character n-gram TF-IDF (2-4 ngrams)      │
│  - Exact core-name dictionary blocking       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 3: 24-Dimensional Feature Engineering │
│  - RapidFuzz token ratios, Jaccard, Winkler  │
│  - Numeric token & postal code overlap       │
│  - Source attribution & composite scores     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 4: Gradient Boosted Trees (LightGBM)  │
│  - Pos-weight scale for extreme imbalance    │
│  - Validation-driven early stopping          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 5: F_0.5 Threshold Optimisation       │
│  - Direct grid sweep targeting macro F_0.5   │
│  - Singleton classification handling         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
     matching_results.tsv  &  candidate_pairs.tsv
```

---

## 📂 Repository Structure

```
.
├── solution/
│   ├── src/
│   │   ├── config.py          # Central hyperparameters and paths
│   │   ├── data_loader.py     # High-speed data loading & splitting
│   │   ├── preprocessor.py    # Text cleaning, suffix & abbreviation logic
│   │   ├── blocker.py         # Multi-strategy CSR candidate generation
│   │   ├── feature_engine.py  # 24 pairwise similarity features
│   │   ├── matcher.py         # LightGBM matcher & threshold tuner
│   │   ├── evaluator.py       # Official F_0.5 metric & recall diagnostics
│   │   └── pipeline.py        # Complete end-to-end orchestration
│   ├── requirements.txt       # Project dependencies
│   └── README.md
├── amazon_ml_challenge_problem_statement.pdf
├── guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r solution/requirements.txt
```

### 2. Running the Complete Pipeline
```bash
python solution/src/pipeline.py
```

### 3. Validating Submission Files
```bash
python student_resource/student_resource/utils/validate_submission.py \
    --matching solution/output/matching_results.tsv \
    --candidate solution/output/candidate_pairs.tsv \
    --test-dir student_resource/student_resource/dataset/test
```

---

## 🔬 Key Engineering Highlights
1. **Sub-linear CSR Dot Products**: Uses sparse matrix memory indexing to scan millions of candidate combinations in seconds with zero memory overhead.
2. **Language-Agnostic Character N-Grams**: Easily handles un-transliterated languages and zero-shot country profiles without rigid rule sets.
3. **Macro $F_{0.5}$ Alignment**: Directly calibrates decision thresholds for precision-weighted evaluation, properly handling singleton vs matched clusters.
