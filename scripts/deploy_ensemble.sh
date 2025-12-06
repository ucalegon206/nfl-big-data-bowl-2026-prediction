#!/bin/bash
# Ensemble Deployment Script for Kaggle
# This script packages and uploads an ensemble to Kaggle

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo "NFL ENSEMBLE DEPLOYMENT TO KAGGLE"
echo "============================================================"
echo ""

# Step 1: Check for ensemble directory
echo "▶ Step 1/5: Checking for ensemble models..."
MODELS_DIR="$REPO_ROOT/models"
ENSEMBLE_DIR=$(ls -d "$MODELS_DIR"/nfl_ensemble_v* 2>/dev/null | sort -r | head -n 1)

if [ -z "$ENSEMBLE_DIR" ]; then
    echo "❌ No ensemble directory found!"
    echo ""
    echo "You need to train the ensemble first:"
    echo "  python scripts/train_ensemble.py --max-rows 200000"
    echo ""
    exit 1
fi

ENSEMBLE_NAME=$(basename "$ENSEMBLE_DIR")
echo "✓ Found ensemble: $ENSEMBLE_NAME"

# Step 2: Create Kaggle package
echo ""
echo "▶ Step 2/5: Creating Kaggle package..."
KAGGLE_PKG_DIR="$REPO_ROOT/for_kaggle"
rm -rf "$KAGGLE_PKG_DIR"
mkdir -p "$KAGGLE_PKG_DIR"

# Copy ensemble directory
cp -r "$ENSEMBLE_DIR" "$KAGGLE_PKG_DIR/"

# Copy features.py
cp "$REPO_ROOT/features.py" "$KAGGLE_PKG_DIR/"

echo "✓ Package created: $KAGGLE_PKG_DIR"

# Step 3: Create dataset metadata
echo ""
echo "▶ Step 3/5: Creating dataset metadata..."

TIMESTAMP=$(basename "$ENSEMBLE_DIR" | sed 's/nfl_ensemble_v//')
# Replace underscores with hyphens for valid slug
DATASET_NAME="nfl-ensemble-v$(echo $TIMESTAMP | sed 's/_/-/g')"

cat > "$KAGGLE_PKG_DIR/dataset-metadata.json" << EOF
{
  "title": "$DATASET_NAME",
  "id": "blazelaserblazer/$DATASET_NAME",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

echo "✓ Metadata created for dataset: $DATASET_NAME"

# Step 4: Upload to Kaggle
echo ""
echo "▶ Step 4/5: Uploading to Kaggle..."
cd "$KAGGLE_PKG_DIR"

if kaggle datasets list --mine | grep -q "$DATASET_NAME"; then
    echo "  Dataset exists, creating new version..."
    kaggle datasets version -p . -m "Ensemble update $(date '+%Y-%m-%d %H:%M:%S')" --dir-mode zip
else
    echo "  Creating new dataset..."
    kaggle datasets create -p . --dir-mode zip
fi

cd "$REPO_ROOT"
echo "✓ Uploaded to Kaggle: $DATASET_NAME"

# Step 5: Update notebook metadata
echo ""
echo "▶ Step 5/5: Updating notebook configuration..."
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
KERNEL_META="$NOTEBOOK_DIR/kernel-metadata.json"

# Read current metadata
CURRENT_META=$(cat "$KERNEL_META")

# Check if dataset already in metadata
if echo "$CURRENT_META" | grep -q "blazelaserblazer/$DATASET_NAME"; then
    echo "  Dataset already in notebook metadata"
else
    echo "  Adding dataset to notebook metadata..."
    # This is a placeholder - user will need to manually add dataset in Kaggle UI
    echo "⚠️  IMPORTANT: Manually add dataset in Kaggle notebook:"
    echo "   1. Go to https://www.kaggle.com/code/blazelaserblazer/nfl-inference"
    echo "   2. Click 'Add data' → Search for '$DATASET_NAME'"
    echo "   3. Click 'Add' and save the notebook"
fi

echo ""
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "Ensemble: $ENSEMBLE_NAME"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Next steps:"
echo "1. Manually add dataset '$DATASET_NAME' to your Kaggle notebook"
echo "2. Push updated notebook: cd notebooks && kaggle kernels push"
echo "3. Test on Kaggle"
echo ""
echo "============================================================"
