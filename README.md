# Amazon ML Challenge 2026 — Multilingual Business Entity Resolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![RapidFuzz](https://img.shields.io/badge/fuzzy--matching-RapidFuzz-orange.svg)](https://github.com/maxbachmann/RapidFuzz)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Submission Status](https://img.shields.io/badge/Submission_Validation-PASSED_(100%25)-success.svg)](#-submission-files--validation-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance Machine Learning & Algorithmic Entity Resolution pipeline built for the **Amazon ML Challenge 2026: Multilingual Business Entity Resolution**.

> [!IMPORTANT]
> **Current Pipeline Status**: Full-scale inference on the entire **1,732,544 test entities** has completed successfully in **7.2 minutes**. The resulting `matching_results.tsv` and `candidate_pairs.tsv` have passed the official Amazon submission validator with **0 errors**.

---

## 📑 Table of Contents
1. [Problem Statement & Dataset Scale](#-problem-statement--dataset-scale)
2. [What Has Been Done (Executive Summary)](#-what-has-been-done-executive-summary)
3. [Architecture & Methodology](#-architecture--methodology)
4. [Submission Files & Validation Status](#-submission-files--validation-status)
5. [Repository Structure](#-repository-structure)
6. [Quickstart Guide (How to Run)](#-quickstart-guide-how-to-run)
7. [Module Reference (`solution/src/`)](#-module-reference-solutionsrc)
8. [Team Onboarding & Next Steps (How to Continue)](#-team-onboarding--next-steps-how-to-continue)

---

## 📌 Problem Statement & Dataset Scale

### The Challenge
Business entity resolution is the task of linking noisy, unstructured business records across heterogeneous data sources that refer to the same real-world entity.

- **Primary Source ($S_1$)**: Target records from multiple jurisdictions (**India, United States, France**).
- **Secondary Sources ($S_2, S_3$)**: Reference records containing duplicate entries, partial matches, and noisy variants.
- **Goal**: For every record in $S_1$, identify:
  1. **Candidate Pool**: A broader set of candidate matches from $S_2 \cup S_3$.
  2. **Matched Entities**: The precise predicted subset of true matches from $S_2 \cup S_3$ (or empty if singleton).

### Dataset Scale & Key Statistics
Our exploratory data analysis (EDA) revealed the following scale:

| Dataset Partition | Source 1 ($S_1$) | Source 2 ($S_2$) | Source 3 ($S_3$) | Total Records |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 2,213,296 | 5,031,780 | 5,284,548 | **~12.53 Million** |
| **Test Set** | **1,732,544** | 4,892,104 | 5,081,248 | **~11.71 Million** |

- **Country Distribution (Test $S_1$)**:
  - 🇮🇳 **India**: 810,317 entities (46.8%)
  - 🇺🇸 **United States**: 663,227 entities (38.3%)
  - 🇫🇷 **France**: 259,000 entities (14.9%)
- **Ground Truth Properties**:
  - **Match Prevalence**: ~94.42% of $S_1$ entities have at least one match in $S_2 \cup S_3$. Only **5.58% are true singletons**.
  - **Match Multiplicity**: Entities with matches link to an average of **3.46 records** (~1.7 in $S_2$ and ~1.8 in $S_3$).

### Evaluation Metric: Macro $F_{0.5}$
The official evaluation metric is the **Macro-averaged $F_{0.5}$ score** across all $S_1$ entities:
$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{i \in S_1} F_{0.5}(P_i, G_i)$$
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

> [!NOTE]
> $F_{0.5}$ weights **precision twice as heavily as recall**. A false positive penalty is substantially higher than a false negative penalty. Predictions must prioritize high-fidelity matching.

---

## 🏆 What Has Been Done (Executive Summary)

1. **Comprehensive Exploratory Data Analysis**: Analyzed link cardinality, singleton ratios, country distributions, string variations, and legal entity abbreviations across English and French corpora.
2. **Modular Production Architecture (`solution/src/`)**: Built separate, testable modules for configuration, data loading, preprocessing, blocking, feature extraction, LightGBM classification, and metric evaluation.
3. **Country-Partitioned Linear-Time Pipeline (`pipeline_production.py`)**:
   - Engineered an ultra-fast, memory-bounded pipeline operating under **5 GB peak RAM**.
   - Resolved all **1,732,544 test entities against ~10M candidates in 433 seconds (~7.2 minutes)** on CPU without out-of-memory crashes.
4. **Validated Submission Outputs**:
   - Generated `solution/output/matching_results.tsv` (374 MB, 1,732,544 rows).
   - Generated `solution/output/candidate_pairs.tsv` (434 MB, 1,732,544 rows).
   - Verified with the official validator: **`PASS — no blocking issues found. Safe to submit.`**

---

## 🏗️ Architecture & Methodology

```
                           Raw TSV Test Data
                   (S1: 1.73M | S2: 4.89M | S3: 5.08M)
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Phase 1: Country-Partitioned Ingestion           │
          │ - Strict partitioning: India / US / France       │
          │ - Zero cross-country false positives             │
          │ - Independent memory reclamation (gc.collect)    │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Phase 2: Multilingual Text Normalization         │
          │ - NFKD Unicode decomposition (strips accents)    │
          │ - Legal suffix removal (LLC, Pvt Ltd, SARL, SAS) │
          │ - Stopword removal across EN and FR              │
          │ - Address number + street anchor extraction      │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Phase 3: Multi-View Inverted Hash Blocking       │
          │ - View 1: Exact Clean Business Name              │
          │ - View 2: Distinctive Core Business Stem         │
          │ - View 3: Compact Core (domain/concatenated)     │
          │ - View 4: Leading Token-Pairs (permutation proof)│
          │ - View 5: Address Street Anchors (geo context)   │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Phase 4: Precision-Calibrated Resolution Rules   │
          │ - Exact core/clean matching                      │
          │ - Compact stem substring matching (len >= 7)     │
          │ - RapidFuzz C++ Token-Sort similarity (>= 88)    │
          │ - Street anchor + token overlap matching         │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Phase 5: Submission Generation & Validation      │
          │ - Guaranteed constraint: Matched ⊆ Candidates    │
          │ - Strict test order preservation                 │
          │ - Tab-separated UTF-8 serialization              │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
           matching_results.tsv        candidate_pairs.tsv
            (1,732,544 rows)            (1,732,544 rows)
```

### Detailed Pipeline Mechanics

#### 1. Country Partitioning
Business entities in this challenge are strictly national; a company in India cannot match an entity registered in France or the US. 
- Processing each country independently reduces candidate space by **~3x to 10x**.
- Memory is released immediately via `gc.collect()` between countries, keeping memory usage capped at **~4.8 GB RAM** instead of overflowing system memory.

#### 2. Multilingual Preprocessing
- **Accents**: Unicode normalization (`NFKD`) strips diacritics (e.g. `Société Générale` $\rightarrow$ `societe generale`).
- **Legal Suffixes**: Over 30 international suffixes are stripped to isolate the true business brand:
  - English: `ltd`, `pvt`, `llc`, `inc`, `corp`, `pllc`, `llp`, `plc`
  - French: `sarl`, `sas`, `sci`, `sa`, `eurl`, `snc`, `groupe`, `societe`
  - German/European: `gmbh`, `ag`, `bv`, `nv`, `spa`, `srl`
- **Address Anchors**: Identifies numeric street numbers + the first primary alphabetical street token (e.g. `123 Main St` $\rightarrow$ `("123", "main")`).

#### 3. 5-View Inverted Hash Blocking
Rather than running an $O(N \times M)$ cross product (~$1.73\text{M} \times 10\text{M} \approx 1.7 \times 10^{13}$ pairs), inverted hash indexes generate candidate pools in **$O(N)$ expected time**:
1. **Exact Clean Name**: Matches full cleaned string.
2. **Core Stem**: Matches distinct core name after suffix stripping.
3. **Compact Core**: Strips all whitespace and punctuation (catches URLs like `amazonpay` vs `amazon pay`).
4. **Token Pairs**: Indexes the first two alphabetical words (resilient to word-order flips).
5. **Address Anchor**: Indexes street number + street token (catches branches sharing an address).

#### 4. Calibrated Decision Thresholds
Because the metric is $F_{0.5}$ (heavily penalizing false positives), matches are only assigned if:
- They share an exact core stem or clean name.
- OR their compact string is an exact substring of length $\ge 7$.
- OR their RapidFuzz `token_sort_ratio` $\ge 88$.
- OR they share high address token overlap ($\ge 4$ common words, or $\ge 3$ words with partial name similarity $\ge 60$).

---

## 📊 Submission Files & Validation Status

The submission files are located in `solution/output/`:

| Output File | Target Rows | File Size | Description |
| :--- | :--- | :--- | :--- |
| `solution/output/matching_results.tsv` | **1,732,544** (+ header) | ~374 MB | `source1_entity_id \t matched_entity_ids` |
| `solution/output/candidate_pairs.tsv` | **1,732,544** (+ header) | ~434 MB | `source1_entity_id \t candidate_entity_ids` |

### Official Validation Run
Validation was performed using the provided script `student_resource/student_resource/utils/validate_submission.py`:

```bash
python student_resource/student_resource/utils/validate_submission.py \
    --matching solution/output/matching_results.tsv \
    --candidate solution/output/candidate_pairs.tsv \
    --test-dir student_resource/student_resource/dataset/test
```

**Validator Output**:
```
Checking row counts...
Checking column names...
Checking for duplicate primary IDs...
Checking for missing primary IDs...
Checking candidate format...
Checking matching format...
Checking candidate-matching consistency...

PASSED:
  Row count matches expected (1,732,544 rows).
  Column names are correct.
  No duplicate or missing source1 entity IDs.
  Format valid (comma-separated IDs without spaces).
  Consistency check passed: all matched IDs are present in candidate IDs.

Status: PASS — no blocking issues found. Safe to submit.
```

---

## 📂 Repository Structure

```
AmazonMLChallange2K26/
├── solution/
│   ├── output/                            # Output submission files
│   │   ├── matching_results.tsv           # Verified matching predictions (374MB)
│   │   └── candidate_pairs.tsv            # Verified candidate pairs (434MB)
│   ├── src/                               # Core Python code
│   │   ├── config.py                      # Hyperparameters, column names, paths
│   │   ├── data_loader.py                 # Streaming TSV parser & validation split
│   │   ├── preprocessor.py                # Multilingual string normalization
│   │   ├── blocker.py                     # CSR TF-IDF & inverted index candidate generator
│   │   ├── feature_engine.py              # 24 pairwise similarity features
│   │   ├── matcher.py                     # LightGBM classifier & F0.5 threshold tuner
│   │   ├── evaluator.py                   # Official Macro F0.5 evaluation implementation
│   │   ├── pipeline.py                    # Modular training + evaluation pipeline
│   │   └── pipeline_production.py         # Ultra-fast end-to-end production test runner
│   ├── requirements.txt                   # Pinned Python dependencies
│   └── README.md                          # Technical solution documentation
├── student_resource/
│   └── student_resource/
│       ├── dataset/
│       │   ├── train/                     # Training sources S1, S2, S3 & ground truth
│       │   └── test/                      # Test sources S1, S2, S3
│       └── utils/
│           ├── evaluation_metric.py       # Official F0.5 metric implementation
│           └── validate_submission.py     # Official submission file validator
├── amazon_ml_challenge_problem_statement.pdf
├── guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf
├── .gitignore
└── README.md                              # Main team project README
```

---

## 🚀 Quickstart Guide (How to Run)

### 1. Environment Setup
Make sure you have Python 3.9+ installed. Install the dependencies:
```bash
pip install -r solution/requirements.txt
```

Key dependencies:
- `rapidfuzz` (C++ accelerated fuzzy string comparisons)
- `lightgbm` (Gradient Boosted Decision Trees)
- `scipy` & `scikit-learn` (Sparse CSR matrices and TF-IDF)
- `pandas` & `numpy`

### 2. Generate Submission Files (Fast Production Pipeline)
To reproduce or regenerate the final test submission files from scratch:
```bash
python solution/src/pipeline_production.py
```
- **Runtime**: ~7.2 minutes.
- **Output files written**:
  - `solution/output/matching_results.tsv`
  - `solution/output/candidate_pairs.tsv`

### 3. Verify Submission with Official Validator
To confirm file validity before submitting:
```bash
python student_resource/student_resource/utils/validate_submission.py \
    --matching solution/output/matching_results.tsv \
    --candidate solution/output/candidate_pairs.tsv \
    --test-dir student_resource/student_resource/dataset/test
```

---

## 🔍 Module Reference (`solution/src/`)

| Script | Purpose & Functionality |
| :--- | :--- |
| [`pipeline_production.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/pipeline_production.py) | **Primary Production Runner**: Streams test TSVs, partitions by country, builds 5-view inverted hash indexes, applies calibrated resolution rules, and outputs verified TSVs. |
| [`config.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/config.py) | **Configuration Hub**: Defines all paths, threshold cutoffs, legal suffix sets, column mappings, and LightGBM hyperparameters. |
| [`data_loader.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/data_loader.py) | **Ingestion**: Efficient generator-based and batch TSV readers for train and test sources; creates stratified validation splits. |
| [`preprocessor.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/preprocessor.py) | **Text Normalization**: Unicode NFKD stripping, regex cleaning, multi-jurisdiction legal suffix pruning, and address standardizer. |
| [`blocker.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/blocker.py) | **Sparse Candidate Generation**: Sparse character n-gram TF-IDF cosine matrix multiplication and inverted dictionary blocking. |
| [`feature_engine.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/feature_engine.py) | **Feature Extraction**: Computes 24 pairwise features per candidate pair (Token Sort Ratio, Partial Ratio, Jaro-Winkler, Jaccard token overlap, Address number match, length ratios). |
| [`matcher.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/matcher.py) | **Model Training & Tuning**: LightGBM binary classifier with `scale_pos_weight` for extreme class imbalance; grid searches probability thresholds specifically targeting Macro $F_{0.5}$. |
| [`evaluator.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/evaluator.py) | **Metric Evaluator**: Exact implementation of Macro $F_{0.5}$, computing precision, recall, and singleton accuracy across all $S_1$ entities. |
| [`pipeline.py`](file:///d:/Sigma/AmazonMLChallange2K26/solution/src/pipeline.py) | **Full ML Pipeline**: Orchestrates training data sampling, feature generation, model training, validation evaluation, and inference. |

---

## 👥 Team Onboarding & Next Steps (How to Continue)

If you are continuing development on this repository, here is how you can jump in and where our biggest opportunities for improvement lie:

### Priority Areas for Iteration

1. **LightGBM Re-scoring on Top Candidates**:
   - Currently, `pipeline_production.py` uses calibrated fuzzy and token overlap rules for speed.
   - You can plug in the trained LightGBM model from `matcher.py` to re-rank the top 10 candidates per entity.
   - Run `python solution/src/pipeline.py` on a sampled training slice (e.g. 50,000 $S_1$ entities) to fit the GBDT weights and test whether GBDT re-ranking improves local validation Macro $F_{0.5}$.

2. **Refining Address Parsing by Country**:
   - **India**: Extract 6-digit postal PIN codes (`r'\b[1-9][0-9]{5}\b'`) and states (Maharashtra, Karnataka, Delhi, etc.).
   - **United States**: Extract 5-digit zip codes (`r'\b\d{5}(?:-\d{4})?\b'`) and 2-letter state abbreviations.
   - **France**: Extract 5-digit French postal codes (`r'\b\d{5}\b'`) and departments.
   - Matching entities with identical postal codes can act as a high-confidence anchor even when names have severe typos.

3. **Graph Clustering / Transitive Closure**:
   - Because $S_2$ and $S_3$ entities are distinct records, if entity $A \in S_1$ matches $B \in S_2$, and $B$ is identical to $C \in S_3$, you can apply connected components or union-find to pull $C$ into the match set for $A$.

4. **Dense Semantic Embeddings (Zero-Shot French / Phonetic Variations)**:
   - For entities with zero lexical overlap, test sentence embeddings using a compact multilingual model (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
   - Use FAISS or Annoy for approximate nearest neighbors on the unresolved singletons (~5.5% of test data).

### Engineering Rules to Keep in Mind
- **Avoid Memory Explosions**: Always partition by country. Never load all 10M records into an unindexed Python list simultaneously.
- **Ensure $Matches \subseteq Candidates$**: The official validator strictly enforces that any entity ID appearing in `matched_entity_ids` must also be in `candidate_entity_ids`.
- **Preserve Output Row Order**: Output files must match the exact row count (1,732,544) and primary ID order of `test_source1.tsv`.
- **Always Validate**: Before submitting, always execute `student_resource/student_resource/utils/validate_submission.py`.
