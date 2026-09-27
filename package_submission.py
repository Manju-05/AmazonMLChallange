"""
package_submission.py — Automated Submission Packager for Amazon ML Challenge 2026.

Creates the official submission archive with the required structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Usage:
    python package_submission.py --team <YourTeamName>
"""
import os
import sys
import argparse
import zipfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "solution", "output")
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
DOC_TEMPLATE = os.path.join(BASE_DIR, "Documentation_template.md")
VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "student_resource", "student_resource", "utils", "validate_submission.py")
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "student_resource", "dataset", "test")
SRC_DIR = os.path.join(BASE_DIR, "solution", "src")
REQ_FILE = os.path.join(BASE_DIR, "solution", "requirements.txt")
SOLUTION_README = os.path.join(BASE_DIR, "solution", "README.md")


def validate_outputs():
    print("=" * 65)
    print("1. Running Official Submission Validator...")
    print("=" * 65)
    if not os.path.exists(MATCHING_FILE):
        print(f"Error: {MATCHING_FILE} not found. Run pipeline_production.py first.")
        sys.exit(1)
    if not os.path.exists(CANDIDATE_FILE):
        print(f"Error: {CANDIDATE_FILE} not found. Run pipeline_production.py first.")
        sys.exit(1)

    cmd = [
        sys.executable,
        VALIDATOR_SCRIPT,
        "--matching", MATCHING_FILE,
        "--candidate", CANDIDATE_FILE,
        "--test-dir", TEST_DIR
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("\nValidator failed! Fix issues before packaging.")
        sys.exit(1)
    print("✓ Validator passed successfully!\n")


def create_zip(team_name: str):
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(BASE_DIR, zip_filename)
    
    print("=" * 65)
    print(f"2. Creating Final Submission Zip: {zip_filename}")
    print("=" * 65)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # 1. output/
        print("  - Adding output/matching_results.tsv ...")
        z.write(MATCHING_FILE, "output/matching_results.tsv")
        print("  - Adding output/candidate_pairs.tsv ...")
        z.write(CANDIDATE_FILE, "output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        print("  - Adding code/business_entity_resolution/README.md ...")
        z.write(SOLUTION_README, "code/business_entity_resolution/README.md")
        
        print("  - Adding code/business_entity_resolution/requirements.txt ...")
        z.write(REQ_FILE, "code/business_entity_resolution/requirements.txt")

        for f in os.listdir(SRC_DIR):
            if f.endswith(".py"):
                fpath = os.path.join(SRC_DIR, f)
                arcname = f"code/business_entity_resolution/src/{f}"
                print(f"  - Adding {arcname} ...")
                z.write(fpath, arcname)

        # 3. Documentation_template.md
        print("  - Adding Documentation_template.md ...")
        z.write(DOC_TEMPLATE, "Documentation_template.md")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print("=" * 65)
    print(f"✓ Submission package created successfully: {zip_path}")
    print(f"  Total Archive Size: {zip_size_mb:.2f} MB")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge submission")
    parser.add_argument("--team", required=True, help="Your team name (used for naming the zip)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running the validator")
    args = parser.parse_args()

    clean_team_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in args.team.strip())
    
    if not args.skip_validation:
        validate_outputs()
    
    create_zip(clean_team_name)


if __name__ == "__main__":
    main()
