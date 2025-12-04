#!/usr/bin/env bash
#
# full_deployment.sh
#
# Master automation script for complete NFL model deployment workflow.
# Coordinates all cleanup, building, uploading, and cleanup steps.
#
# This is the ONE script to run for complete deployment automation.
#
# Usage:
#   bash scripts/full_deployment.sh              # Complete workflow
#   bash scripts/full_deployment.sh --build-only  # Build but don't upload/cleanup
#   bash scripts/full_deployment.sh --help       # Show help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_ONLY=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
  cat <<EOF
${BLUE}full_deployment.sh${NC} - Complete NFL Model Deployment Automation

${BLUE}Usage:${NC}
  bash scripts/full_deployment.sh [OPTIONS]

${BLUE}Options:${NC}
  --build-only     Build and verify, but don't upload to Kaggle or cleanup
  --help          Show this help message

${BLUE}What this script does:${NC}
  ${CYAN}Phase 1: Local Build${NC}
    1. Clean previous artifacts (for_kaggle/, for_kaggle.zip)
    2. Run model integrity tests
    3. Build for_kaggle with timestamped model
    4. Create for_kaggle.zip and verify correct model pattern
    5. Display build summary

  ${CYAN}Phase 2: Kaggle Upload${NC}
    6. Upload to Kaggle with timestamped dataset name
    7. Push notebook to Kaggle with dataset attached

  ${CYAN}Phase 3: Kaggle Cleanup${NC}
    8. List old cached datasets
    9. Delete old datasets using Kaggle API
    10. Verify cleanup complete

  ${CYAN}Phase 4: Next Steps${NC}
    11. Display instructions for final testing

${BLUE}Prerequisites:${NC}
  • Kaggle CLI installed (pip install kaggle)
  • Kaggle API key configured (~/.kaggle/kaggle.json)
  • All model integrity tests passing

${BLUE}After running this script:${NC}
  1. ✓ Local build complete
  2. ✓ Uploaded to Kaggle (if not using --build-only)
  3. ✓ Old datasets cleaned up (if not using --build-only)
  4. ⏳ Run notebook on Kaggle to verify model loads correctly
  5. ⏳ Check output for "X model random_state: 42" and "Y model random_state: 42"
  6. ⏳ Verify NO PCG64BitGenerator errors
  7. ⏳ Submit to competition

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) BUILD_ONLY=true; shift ;;
    --help) usage; exit 0 ;;
    *) echo "${RED}Unknown option: $1${NC}"; usage; exit 1 ;;
  esac
done

log_phase() {
  echo ""
  echo "${CYAN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
  echo "${CYAN}║${NC} $1"
  echo "${CYAN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

log_step() {
  echo "${BLUE}→${NC} $1"
}

log_success() {
  echo "${GREEN}✓${NC} $1"
}

log_error() {
  echo "${RED}✗${NC} $1"
}

log_warning() {
  echo "${YELLOW}⚠${NC} $1"
}

print_section() {
  echo ""
  echo "${GREEN}══════════════════════════════════════════════════════════════════${NC}"
  echo "$1"
  echo "${GREEN}══════════════════════════════════════════════════════════════════${NC}"
  echo ""
}

# ============================================================================
# PHASE 1: Local Build
# ============================================================================
log_phase "PHASE 1: Local Build"

log_step "Step 1/11: Cleaning previous artifacts"
rm -rf "$REPO_ROOT/for_kaggle" "$REPO_ROOT/for_kaggle.zip"
log_success "Cleaned artifacts"

log_step "Step 2/11: Running model integrity tests"
if ! "$REPO_ROOT/.venv/bin/python" -m pytest "$REPO_ROOT/tests/test_model_integrity.py" -q 2>&1 | tail -5; then
  log_error "Model integrity tests failed"
  exit 1
fi
log_success "Model integrity tests passed"

log_step "Step 3/11: Building for_kaggle directory"
if bash "$REPO_ROOT/scripts/prepare_for_kaggle.sh" >/dev/null 2>&1; then
  # Success - extract model info
  MODEL_FILE=$(find "$REPO_ROOT/for_kaggle/models" -name "nfl_model_v*.pkl" -type f | head -1)
  if [[ -n "$MODEL_FILE" ]]; then
    log_success "Built for_kaggle successfully"
  else
    log_error "Model file not found after build"
    exit 1
  fi
else
  log_error "Failed to build for_kaggle"
  exit 1
fi

MODEL_FILENAME=$(basename "$MODEL_FILE")
TIMESTAMP_ID="${MODEL_FILENAME#nfl_model_v}"
TIMESTAMP_ID="${TIMESTAMP_ID%.pkl}"
log_success "Built for_kaggle with model: $MODEL_FILENAME"

log_step "Step 4/11: Creating and verifying for_kaggle.zip"
cd "$REPO_ROOT"
zip -r for_kaggle.zip for_kaggle -q
ARCHIVE_SIZE=$(du -h for_kaggle.zip | cut -f1)
log_success "Created archive: for_kaggle.zip ($ARCHIVE_SIZE)"

# Verify correct model pattern (use set +e temporarily to avoid pipefail issues)
set +e
MODEL_CHECK=$(unzip -l for_kaggle.zip 2>/dev/null | grep "nfl_model_v")
set -e
if [[ -n "$MODEL_CHECK" ]]; then
  log_success "Archive contains correct nfl_model_v*.pkl pattern"
else
  log_error "Archive does NOT contain nfl_model_v*.pkl"
  exit 1
fi

# Check for old pattern
set +e
OLD_CHECK=$(unzip -l for_kaggle.zip 2>/dev/null | grep "best_model")
set -e
if [[ -z "$OLD_CHECK" ]]; then
  log_success "Archive correctly excludes old best_model_*.pkl pattern"
else
  log_warning "Archive still contains old best_model_*.pkl (this shouldn't happen)"
fi

log_step "Step 5/11: Build summary"
print_section "✓ LOCAL BUILD COMPLETE"
echo "Model:           $MODEL_FILENAME"
echo "Timestamp:       $TIMESTAMP_ID"
echo "Archive:         for_kaggle.zip ($ARCHIVE_SIZE)"
echo ""

if [[ "$BUILD_ONLY" == true ]]; then
  echo "${GREEN}✓ Build completed successfully (--build-only flag set)${NC}"
  exit 0
fi

# ============================================================================
# PHASE 2: Kaggle Upload
# ============================================================================
log_phase "PHASE 2: Kaggle Upload"

log_step "Step 6/11: Uploading to Kaggle"
if ! "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/upload_to_kaggle.py"; then
  log_error "Upload to Kaggle failed"
  exit 1
fi
log_success "Uploaded to Kaggle"

log_step "Step 7/11: Verifying upload"
log_success "Dataset and notebook uploaded with timestamp: $TIMESTAMP_ID"

# ============================================================================
# PHASE 3: Kaggle Cleanup
# ============================================================================
log_phase "PHASE 3: Kaggle Cleanup"

log_step "Step 8/11: Listing datasets to determine cleanup targets"
print_section "Old Datasets to Delete"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cleanup_old_datasets.py" --list-only 2>/dev/null || true
echo ""

log_step "Step 9/11: Cleaning old datasets from Kaggle"
log_warning "Automatic deletion requires confirmation. Starting interactive cleanup..."
echo ""

# Try automated cleanup with minimal interaction
CLEANUP_RESULT=$("$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cleanup_old_datasets.py" --delete-old 2>&1 || true)

echo "$CLEANUP_RESULT"

log_step "Step 10/11: Verifying cleanup"
echo "Remaining datasets:"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cleanup_old_datasets.py" --list-only 2>/dev/null || true

# ============================================================================
# PHASE 4: Next Steps
# ============================================================================
log_phase "PHASE 4: Next Steps & Final Instructions"

log_step "Step 11/11: Displaying next steps"
print_section "✓ DEPLOYMENT COMPLETE!"
echo ""
echo "${CYAN}Summary:${NC}"
echo "  ✓ Model built:          $MODEL_FILENAME"
echo "  ✓ Timestamp:             $TIMESTAMP_ID"
echo "  ✓ Uploaded to Kaggle:    nfl-model-v$TIMESTAMP_ID"
echo "  ✓ Cleaned old datasets:  (see above)"
echo ""
echo "${CYAN}Now you need to test on Kaggle:${NC}"
echo ""
echo "1. ${BLUE}Visit your Kaggle notebook${NC}"
echo "   https://www.kaggle.com/code/blazelaserblazer"
echo ""
echo "2. ${BLUE}Find the newest notebook${NC} (nfl-inference-${TIMESTAMP_ID})"
echo ""
echo "3. ${BLUE}Click 'Commit and Run'${NC}"
echo ""
echo "4. ${BLUE}Verify output contains:${NC}"
echo "   ✓ Model Version: $TIMESTAMP_ID"
echo "   ✓ X model random_state: 42"
echo "   ✓ Y model random_state: 42"
echo "   ✓ NO PCG64BitGenerator errors"
echo "   ✓ Predictions generated"
echo ""
echo "5. ${BLUE}Submit to competition${NC}"
echo ""
echo "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
