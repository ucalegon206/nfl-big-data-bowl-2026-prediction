#!/usr/bin/env bash
#
# clean_and_deploy.sh
#
# Automated deployment workflow that:
# 1. Cleans previous artifacts (for_kaggle, for_kaggle.zip)
# 2. Builds fresh for_kaggle directory with timestamped model
# 3. Creates for_kaggle.zip and verifies correct model pattern
# 4. Uploads to Kaggle with timestamped dataset name
# 5. Pushes notebook to Kaggle
# 6. Provides instructions for cleanup and testing
#
# Usage:
#   bash scripts/clean_and_deploy.sh              # Use default timestamps
#   bash scripts/clean_and_deploy.sh --skip-upload  # Build only, don't upload
#   bash scripts/clean_and_deploy.sh --help       # Show help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/for_kaggle"
SKIP_UPLOAD=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
  cat <<EOF
${BLUE}clean_and_deploy.sh${NC} - Automated Kaggle deployment workflow

${BLUE}Usage:${NC}
  bash scripts/clean_and_deploy.sh [OPTIONS]

${BLUE}Options:${NC}
  --skip-upload    Build for_kaggle and zip, but don't upload to Kaggle
  --help          Show this help message

${BLUE}What this script does:${NC}
  1. Clean previous artifacts (for_kaggle/, for_kaggle.zip)
  2. Run model integrity tests
  3. Build for_kaggle with timestamped model (nfl_model_v{YYYYMMDD_HHMMSS}.pkl)
  4. Create for_kaggle.zip and verify correct model pattern
  5. Upload to Kaggle with full timestamp in dataset name
  6. Push notebook to Kaggle
  7. Display cleanup and testing instructions

${BLUE}Prerequisites:${NC}
  - Kaggle CLI installed (pip install kaggle)
  - Kaggle API key configured (~/.kaggle/kaggle.json)
  - All model integrity tests passing

${BLUE}After deployment, you must:${NC}
  1. Delete old datasets from https://www.kaggle.com/settings/datasets
     (Datasets without the new timestamp pattern)
  2. Run the notebook on Kaggle to verify it loads the correct model
  3. Check for "X model random_state: 42" and "Y model random_state: 42" in output
  4. Verify NO PCG64BitGenerator errors appear
  5. Submit to the competition

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-upload) SKIP_UPLOAD=true; shift ;;
    --help) usage; exit 0 ;;
    *) echo "${RED}Unknown option: $1${NC}"; usage; exit 1 ;;
  esac
done

log_section() {
  echo ""
  echo "${BLUE}=================================================================================${NC}"
  echo "${BLUE}$1${NC}"
  echo "${BLUE}=================================================================================${NC}"
  echo ""
}

log_step() {
  echo "${GREEN}▶${NC} $1"
}

log_success() {
  echo "${GREEN}✓${NC} $1"
}

log_error() {
  echo "${RED}✗${NC} $1"
}

log_info() {
  echo "${BLUE}ℹ${NC} $1"
}

log_warning() {
  echo "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# STEP 1: Clean previous artifacts
# ============================================================================
log_section "STEP 1: Cleaning Previous Artifacts"
log_step "Removing for_kaggle/ and for_kaggle.zip"
rm -rf "$OUT" "$REPO_ROOT/for_kaggle.zip"
log_success "Cleaned previous artifacts"

# ============================================================================
# STEP 2: Run model integrity tests
# ============================================================================
log_section "STEP 2: Model Integrity Tests"
log_step "Running pytest on model integrity tests..."
if ! "$REPO_ROOT/.venv/bin/python" -m pytest "$REPO_ROOT/tests/test_model_integrity.py" -v 2>&1 | tail -20; then
  log_error "Model integrity tests failed!"
  exit 1
fi
log_success "All model integrity tests passed"

# ============================================================================
# STEP 3: Build for_kaggle with timestamped model
# ============================================================================
log_section "STEP 3: Building for_kaggle Directory"
log_step "Running prepare_for_kaggle.sh to build fresh for_kaggle..."
if ! bash "$REPO_ROOT/scripts/prepare_for_kaggle.sh"; then
  log_error "Failed to build for_kaggle"
  exit 1
fi
log_success "for_kaggle directory built successfully"

# Extract the actual model filename that was created
MODEL_FILE=$(find "$OUT/models" -name "nfl_model_v*.pkl" -type f | head -1)
if [[ -z "$MODEL_FILE" ]]; then
  log_error "Model file not found in $OUT/models"
  exit 1
fi
MODEL_FILENAME=$(basename "$MODEL_FILE")
log_info "Created model: $MODEL_FILENAME"

# ============================================================================
# STEP 4: Create for_kaggle.zip and verify model pattern
# ============================================================================
log_section "STEP 4: Creating and Verifying for_kaggle.zip"
log_step "Creating for_kaggle.zip archive..."
cd "$REPO_ROOT"
zip -r for_kaggle.zip for_kaggle -q
log_success "Archive created: for_kaggle.zip"

log_step "Verifying archive contains nfl_model_v*.pkl (NEW pattern)..."
ARCHIVE_MODEL=$(unzip -l for_kaggle.zip | grep "nfl_model_v.*\.pkl")
if [[ -n "$ARCHIVE_MODEL" ]]; then
  log_success "✓ Archive contains correct model pattern"
  log_info "Model in archive: $(echo "$ARCHIVE_MODEL" | grep -oE "nfl_model_v[^[:space:]]*\.pkl")"
else
  log_error "Archive does NOT contain nfl_model_v*.pkl pattern"
  exit 1
fi

log_step "Verifying archive does NOT contain best_model_*.pkl (OLD pattern)..."
OLD_PATTERN=$(unzip -l for_kaggle.zip | grep "best_model_.*\.pkl" || true)
if [[ -n "$OLD_PATTERN" ]]; then
  log_warning "Archive still contains OLD pattern (best_model_*.pkl) - this should not happen"
  log_warning "Found: $(echo "$OLD_PATTERN" | grep -oE "best_model[^[:space:]]*\.pkl")"
else
  log_success "✓ Archive correctly excludes old best_model_*.pkl pattern"
fi

log_step "Archive verification complete"
ls -lh for_kaggle.zip

# ============================================================================
# STEP 5: Upload to Kaggle (conditional)
# ============================================================================
if [[ "$SKIP_UPLOAD" == true ]]; then
  log_section "SKIPPING UPLOAD (--skip-upload flag set)"
  log_info "Build completed successfully"
  log_info "Archive ready: for_kaggle.zip ($(du -h for_kaggle.zip | cut -f1))"
  log_info "Run again without --skip-upload to proceed with Kaggle upload"
else
  log_section "STEP 5: Uploading to Kaggle"
  log_step "Starting dataset and notebook upload..."
  
  if ! "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/upload_to_kaggle.py"; then
    log_error "Upload to Kaggle failed"
    exit 1
  fi
  
  log_success "Upload to Kaggle completed"
  
  # ============================================================================
  # STEP 6: Post-deployment instructions
  # ============================================================================
  log_section "STEP 6: Post-Deployment Instructions"
  
  log_warning "CRITICAL: Old cached datasets must be deleted"
  echo ""
  echo "1. ${YELLOW}Delete old datasets from Kaggle${NC}"
  echo "   Visit: https://www.kaggle.com/settings/datasets"
  echo "   Delete any datasets matching these patterns:"
  echo "     • nfl-model-v20* (without time component, e.g., nfl-model-v20251204)"
  echo "     • nfl-model-v20*-* (with old timestamp, e.g., nfl-model-v20251204-100817)"
  echo "   Keep ONLY the newest: nfl-model-v${MODEL_FILENAME#*_v} (with full timestamp)"
  echo ""
  
  log_warning "IMPORTANT: Test on Kaggle"
  echo "2. Run the notebook on Kaggle"
  echo "   Visit: https://www.kaggle.com/code/blazelaserblazer"
  echo "   Find: nfl-inference-* (the newest one with full timestamp)"
  echo "   Click 'Commit and Run' button"
  echo ""
  
  log_info "Verify output:"
  echo "   ✓ Shows 'LOADING MODEL AND FEATURES'"
  echo "   ✓ Shows 'Model Version: [timestamp]'"
  echo "   ✓ Shows 'X model random_state: 42'"
  echo "   ✓ Shows 'Y model random_state: 42'"
  echo "   ✓ NO errors about PCG64BitGenerator"
  echo "   ✓ NO errors about missing model files"
  echo "   ✓ Produces predictions.csv"
  echo ""
  
  log_success "Once verified, submit to competition!"
  
  log_section "DEPLOYMENT COMPLETE"
  log_success "All automated steps completed successfully"
  log_info "Model: $MODEL_FILENAME"
  log_info "Archive: for_kaggle.zip ($(du -h for_kaggle.zip | cut -f1))"
  echo ""
fi

# Display summary
log_section "DEPLOYMENT SUMMARY"
echo ""
echo "✓ Step 1: Cleaned artifacts"
echo "✓ Step 2: Model integrity tests passed"
echo "✓ Step 3: Built for_kaggle with fresh timestamp"
echo "✓ Step 4: Created and verified for_kaggle.zip"
if [[ "$SKIP_UPLOAD" != true ]]; then
  echo "✓ Step 5: Uploaded to Kaggle (dataset + notebook)"
  echo "✓ Step 6: Displayed post-deployment instructions"
fi
echo ""
echo "${GREEN}Ready for Kaggle deployment!${NC}"
echo ""
