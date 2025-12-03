#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/for_kaggle"
ZIP=false
COMMIT=false

usage(){
  cat <<EOF
Usage: $0 [--zip] [--commit]

Options:
  --zip       Create for_kaggle.zip archive after building folder
  --commit    Add, commit and push the generated files to git (runs git add + commit + push)
  --help      Show this help

This script collects the minimal files required to run the inference notebook on Kaggle
and places them in the `for_kaggle` directory at the repository root (or creates a zip).
Files copied:
 - models/best_model.pkl
 - test.csv, test_input.csv
 - scripts/ (all files)
 - features.py
 - KAGGLE_SUBMIT.md
 - notebooks/inference_only_submission.ipynb
 - notebooks/submission_notebook.ipynb
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP=true; shift ;;
    --commit) COMMIT=true; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

echo "Preparing for_kaggle dataset in: $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

# Helper to copy if exists
copy_if_exists(){
  src="$1"
  dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    echo "copied: $src -> $dst"
  else
    echo "warning: not found: $src"
  fi
}

# Copy model
copy_if_exists "$REPO_ROOT/models/best_model.pkl" "$OUT/models/best_model.pkl"

# Copy test files
copy_if_exists "$REPO_ROOT/test.csv" "$OUT/test.csv"
copy_if_exists "$REPO_ROOT/test_input.csv" "$OUT/test_input.csv"

# Copy scripts directory (exclude __pycache__)
if [ -d "$REPO_ROOT/scripts" ]; then
  mkdir -p "$OUT/scripts"
  rsync -av --exclude='__pycache__' --exclude='*.pyc' "$REPO_ROOT/scripts/" "$OUT/scripts/" >/dev/null
  echo "copied: scripts/ -> $OUT/scripts/"
else
  echo "warning: scripts/ not found"
fi

# Copy top-level feature file(s)
copy_if_exists "$REPO_ROOT/features.py" "$OUT/features.py"
copy_if_exists "$REPO_ROOT/KAGGLE_SUBMIT.md" "$OUT/KAGGLE_SUBMIT.md"

# Copy notebooks (inference-only and main submission notebook)
copy_if_exists "$REPO_ROOT/notebooks/inference_only_submission.ipynb" "$OUT/inference_only_submission.ipynb"
copy_if_exists "$REPO_ROOT/notebooks/submission_notebook.ipynb" "$OUT/submission_notebook.ipynb"

# Clean up any .git or large folders unintentionally copied
if [ -d "$OUT/.git" ]; then
  rm -rf "$OUT/.git"
fi

# Optionally create zip
if [ "$ZIP" = true ]; then
  (cd "$REPO_ROOT" && rm -f for_kaggle.zip && zip -r for_kaggle.zip for_kaggle >/dev/null)
  echo "Created archive: $REPO_ROOT/for_kaggle.zip"
fi

# Optionally git commit and push
if [ "$COMMIT" = true ]; then
  echo "Committing generated files to git..."
  (cd "$REPO_ROOT" && git add for_kaggle $( [ "$ZIP" = true ] && echo for_kaggle.zip || true ) && git commit -m "Add generated for_kaggle dataset" || true)
  (cd "$REPO_ROOT" && git push origin HEAD)
  echo "Committed and pushed"
fi

echo "Done. for_kaggle content:" 
ls -la "$OUT" || true
