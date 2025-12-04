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

echo "Running model integrity tests..."
if ! "$REPO_ROOT/.venv/bin/python" -m pytest "$REPO_ROOT/tests/test_model_integrity.py" -v; then
  echo "ERROR: Model integrity tests failed!"
  echo "Cannot prepare for Kaggle with a broken model."
  exit 1
fi
echo "✓ Model integrity tests passed"
echo ""

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

# Copy model with NEW filename pattern to defeat Kaggle caching
# Use nfl_model_v{timestamp}.pkl instead of best_model_{timestamp}.pkl
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
READABLE_TS=$(date +"%Y-%m-%d %H:%M:%S")
if [ -f "$REPO_ROOT/models/best_model.pkl" ]; then
  mkdir -p "$OUT/models"
  
  # Create new versioned model with DIFFERENT name pattern
  VERSIONED_MODEL="$OUT/models/nfl_model_v${TIMESTAMP}.pkl"
  cp -a "$REPO_ROOT/models/best_model.pkl" "$VERSIONED_MODEL"
  echo "copied (versioned): $REPO_ROOT/models/best_model.pkl -> $VERSIONED_MODEL"
  echo "✓ Using versioned model: nfl_model_v${TIMESTAMP}.pkl"
  echo "✓ Timestamp: $READABLE_TS"
  echo "✓ NEW naming pattern to defeat Kaggle caching!"
  
  # Create metadata file for the model
  MODEL_METADATA="$OUT/models/MODEL_METADATA.txt"
  cat > "$MODEL_METADATA" << EOF
Model Metadata
==============
Filename: nfl_model_v${TIMESTAMP}.pkl
Created: $READABLE_TS
Timestamp ID: ${TIMESTAMP}

This model package contains:
- Two HistGradientBoostingRegressor models (x and y coordinates)
- random_state set to 42 (integer, not RandomState object)
- Feature columns list for inference
- Player position values for feature engineering

Deployment Notes:
- Model uses nfl_model_v{timestamp}.pkl naming to defeat Kaggle caching
- When uploading to Kaggle, include this metadata in dataset
- Ensure NumPy compatibility by checking random_state is integer 42

See KAGGLE_SUBMIT.md for submission instructions.
EOF
  echo "✓ Created model metadata at: $MODEL_METADATA"
  
  # Copy the last 3 existing versioned models from models/ directory (if they exist)
  # Check both old and new patterns
  EXISTING_VERSIONED=$(ls -t "$REPO_ROOT/models"/nfl_model_v[0-9]*.pkl "$REPO_ROOT/models"/best_model_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].pkl 2>/dev/null | head -3 || true)
  if [ -n "$EXISTING_VERSIONED" ]; then
    echo "Including previous versioned models:"
    for prev_model in $EXISTING_VERSIONED; do
      prev_basename=$(basename "$prev_model")
      # Don't duplicate if we just created it
      if [ "$prev_basename" != "nfl_model_v${TIMESTAMP}.pkl" ]; then
        cp -a "$prev_model" "$OUT/models/$prev_basename"
        echo "  copied: $prev_basename"
      fi
    done
  fi
else
  echo "ERROR: models/best_model.pkl not found"
  exit 1
fi

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
ls -la "$OUT" 2>/dev/null || echo "  (directory listing skipped)"
exit 0
