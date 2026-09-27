"""
validate_and_package.py - Run after pipeline completes.
Steps:
1. Validate output files with official validator
2. Show match statistics
3. Create submission package zip
"""
import subprocess, sys, os

BASE = os.path.abspath('.')
MATCHING = os.path.join(BASE, 'solution', 'output', 'matching_results.tsv')
CANDIDATE = os.path.join(BASE, 'solution', 'output', 'candidate_pairs.tsv')
VALIDATOR = os.path.join(BASE, 'student_resource', 'student_resource', 'utils', 'validate_submission.py')
TEST_DIR = os.path.join(BASE, 'student_resource', 'student_resource', 'dataset', 'test')

# Check files exist
for f in [MATCHING, CANDIDATE]:
    if not os.path.exists(f):
        print(f"ERROR: {f} does not exist. Run pipeline first.")
        sys.exit(1)
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")

# Quick stats on matching_results.tsv
print("\nComputing match statistics...")
total = 0
matched = 0
total_links = 0
with open(MATCHING, 'r', encoding='utf-8') as f:
    next(f)  # header
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 2:
            total += 1
            ids = parts[1].strip()
            if ids:
                matched += 1
                total_links += len(ids.split(','))

print(f"  Total S1 entities: {total:,}")
print(f"  Entities with matches: {matched:,} ({100*matched/total:.1f}%)")
print(f"  Singletons: {total-matched:,} ({100*(total-matched)/total:.1f}%)")
print(f"  Total match links: {total_links:,}")
print(f"  Avg links per matched entity: {total_links/max(matched,1):.2f}")

# Run official validator
print("\n" + "="*60)
print("Running official validator...")
print("="*60)
result = subprocess.run(
    [sys.executable, VALIDATOR,
     '--matching', MATCHING,
     '--candidate', CANDIDATE,
     '--test-dir', TEST_DIR],
    capture_output=False
)
if result.returncode == 0:
    print("\nVALIDATION PASSED! Ready to submit.")
else:
    print("\nVALIDATION FAILED. Fix issues before submitting.")
    sys.exit(1)
