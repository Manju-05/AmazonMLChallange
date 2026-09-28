"""
config.py — Central configuration for the Entity Resolution pipeline.
All paths, constants, and hyperparameters in one place.
"""
import os

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "student_resource", "student_resource", "dataset")

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")

TRAIN_S1   = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2   = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3   = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT   = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1    = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2    = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3    = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCHING_OUTPUT    = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT   = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ──────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────

# Legal suffix normalisation mapping (short → full)
LEGAL_SUFFIX_MAP = {
    # English
    "pvt":          "private",
    "pvt.":         "private",
    "ltd":          "limited",
    "ltd.":         "limited",
    "llc":          "llc",
    "llc.":         "llc",
    "inc":          "incorporated",
    "inc.":         "incorporated",
    "corp":         "corporation",
    "corp.":        "corporation",
    "co":           "company",
    "co.":          "company",
    "pllc":         "pllc",
    "pllc.":        "pllc",
    "llp":          "llp",
    "llp.":         "llp",
    "plc":          "plc",
    "plc.":         "plc",
    # French
    "sarl":         "sarl",
    "s.a.r.l":      "sarl",
    "s.a.r.l.":     "sarl",
    "sas":          "sas",
    "s.a.s":        "sas",
    "s.a.s.":       "sas",
    "sci":          "sci",
    "s.c.i":        "sci",
    "s.c.i.":       "sci",
    "sa":           "sa",
    "s.a":          "sa",
    "s.a.":         "sa",
    "eurl":         "eurl",
    "snc":          "snc",
    "société":      "societe",
    "groupe":       "groupe",
    # Indian
    "m/s":          "",
    "shri":         "shri",
}

# Tokens to REMOVE entirely from core name extraction
LEGAL_SUFFIX_TOKENS = {
    "private", "limited", "llc", "incorporated", "corporation",
    "company", "pllc", "llp", "plc", "pvt", "ltd", "inc", "corp",
    "co", "sarl", "sas", "sci", "sa", "eurl", "snc", "groupe",
    "societe", "m/s", "pvt.", "ltd.", "inc.", "corp.", "co.",
    "llc.", "pllc.", "llp.", "plc.",
}

# Address abbreviation expansions
ADDR_ABBREVIATIONS = {
    # US street types
    "st":    "street",
    "st.":   "street",
    "rd":    "road",
    "rd.":   "road",
    "ave":   "avenue",
    "ave.":  "avenue",
    "blvd":  "boulevard",
    "blvd.": "boulevard",
    "dr":    "drive",
    "dr.":   "drive",
    "ln":    "lane",
    "ln.":   "lane",
    "ct":    "court",
    "ct.":   "court",
    "cir":   "circle",
    "cir.":  "circle",
    "pl":    "place",
    "pl.":   "place",
    "pkwy":  "parkway",
    "hwy":   "highway",
    "apt":   "apartment",
    "ste":   "suite",
    "bldg":  "building",
    # French
    "r.":    "rue",
    "bd":    "boulevard",
    "bd.":   "boulevard",
    "av":    "avenue",
    "av.":   "avenue",
    # Indian
    "opp":   "opposite",
    "opp.":  "opposite",
    "nr":    "near",
    "nr.":   "near",
}

# ──────────────────────────────────────────────────────────
# Blocking Hyperparameters
# ──────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 100_000      # vocabulary size for TF-IDF
TFIDF_NGRAM_RANGE   = (2, 4)       # character n-grams for language-agnostic matching
BLOCKING_TOP_K      = 25           # top-K candidates per blocking strategy
BLOCKING_MIN_SCORE  = 0.20         # minimum TF-IDF similarity to consider

# ──────────────────────────────────────────────────────────
# Model & Evaluation
# ──────────────────────────────────────────────────────────
VALIDATION_SPLIT       = 0.2          # fraction of S1 entities for validation
TRAIN_SAMPLE_ENTITIES  = 50_000       # representative training entities (~500k pairs for LightGBM)
VAL_SAMPLE_ENTITIES    = 10_000       # validation entities for fast, precise threshold tuning
RANDOM_SEED            = 42
F_BETA                 = 0.5          # F_0.5 metric
THRESHOLD_SWEEP        = [i/100 for i in range(10, 95, 1)]  # 0.10 to 0.94

# ──────────────────────────────────────────────────────────
# Resource Management
# ──────────────────────────────────────────────────────────
BATCH_SIZE          = 50_000       # batch size for large operations
MAX_WORKERS         = 4            # parallelism
