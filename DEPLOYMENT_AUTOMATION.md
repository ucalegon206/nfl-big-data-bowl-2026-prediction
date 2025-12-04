# Deployment & Cleanup Automation

Complete automation suite for NFL model deployment workflow. This documentation covers all cleanup and deployment scripts.

## Quick Start

Run the **complete end-to-end deployment**:

```bash
bash scripts/full_deployment.sh
```

This single command handles:
1. ✓ Clean previous artifacts
2. ✓ Model integrity tests
3. ✓ Build for_kaggle with fresh timestamp
4. ✓ Create and verify for_kaggle.zip
5. ✓ Upload to Kaggle
6. ✓ Clean old datasets
7. ✓ Display next steps

## Available Scripts

### 1. `full_deployment.sh` - Master Orchestration

**Purpose:** One-command complete deployment workflow

**Usage:**
```bash
# Complete workflow (build + upload + cleanup)
bash scripts/full_deployment.sh

# Build only, don't upload or cleanup
bash scripts/full_deployment.sh --build-only

# Show help
bash scripts/full_deployment.sh --help
```

**What it does:**
- **Phase 1:** Local Build
  - Cleans artifacts
  - Runs model tests
  - Builds for_kaggle
  - Verifies model pattern in archive

- **Phase 2:** Kaggle Upload
  - Uploads timestamped dataset
  - Pushes notebook with dataset attached

- **Phase 3:** Kaggle Cleanup
  - Lists old cached datasets
  - Deletes old datasets via API

- **Phase 4:** Next Steps
  - Displays testing instructions

**Exit Status:**
- `0` = Success
- `1` = Failure (check error messages)

---

### 2. `clean_and_deploy.sh` - Build & Upload Only

**Purpose:** Build and upload without automatic cleanup

**Usage:**
```bash
# Full build and upload
bash scripts/clean_and_deploy.sh

# Build only, skip upload
bash scripts/clean_and_deploy.sh --skip-upload

# Show help
bash scripts/clean_and_deploy.sh --help
```

**Features:**
- Cleaner output with color codes
- Detailed verification of model pattern
- Skippable upload step
- Manual cleanup instructions printed

**Output:**
- Step-by-step progress messages
- Color-coded success/warning/error indicators
- Post-deployment checklist

---

### 3. `cleanup_old_datasets.py` - Kaggle Dataset Cleanup

**Purpose:** Identify and delete old cached datasets from Kaggle

**Usage:**
```bash
# List old datasets (safe, no deletion)
python scripts/cleanup_old_datasets.py --list-only

# Delete old datasets (interactive, requires confirmation)
python scripts/cleanup_old_datasets.py --delete-old

# Show help
python scripts/cleanup_old_datasets.py --help
```

**How it identifies old datasets:**

| Pattern | Type | Keep/Delete |
|---------|------|-------------|
| `nfl-model-v20251204` | No timestamp | ❌ Delete |
| `nfl-model-v20251204-100817` | Old time format (HHMM) | ❌ Delete |
| `nfl-model-v20251204-101918` | New format (HHMMSS) | ✓ Keep |

**Workflow:**
1. Lists all nfl-model-v* datasets
2. Categorizes into old/new based on timestamp format
3. Shows what will be deleted
4. Requires `yes` confirmation before deletion
5. Deletes via Kaggle API
6. Confirms cleanup

**Safety Features:**
- `--list-only` (default) is safe, no deletion
- Requires explicit confirmation for deletion
- Shows datasets before and after cleanup
- Handles API failures gracefully

---

### 4. `prepare_for_kaggle.sh` - Prepare Directory

**Purpose:** Validate and package model for deployment

**Usage:**
```bash
# Standard preparation
bash scripts/prepare_for_kaggle.sh

# With archiving
bash scripts/prepare_for_kaggle.sh --zip

# With git commit
bash scripts/prepare_for_kaggle.sh --commit

# Combined
bash scripts/prepare_for_kaggle.sh --zip --commit
```

**What it does:**
- Runs all model integrity tests
- Creates for_kaggle directory
- Copies files with proper structure
- Creates MODEL_METADATA.txt with timestamp
- Optionally creates zip archive
- Optionally commits to git

**Output:**
- Timestamped model: `nfl_model_v{YYYYMMDD_HHMMSS}.pkl`
- Metadata file with audit trail
- Success/failure indicators

---

### 5. `upload_to_kaggle.py` - Upload to Kaggle

**Purpose:** Upload dataset and notebook to Kaggle

**Usage:**
```bash
# Use default names (with current timestamp)
python scripts/upload_to_kaggle.py

# Custom dataset name
python scripts/upload_to_kaggle.py --dataset-name my-dataset

# Custom notebook name
python scripts/upload_to_kaggle.py --notebook-name my-notebook

# Show help
python scripts/upload_to_kaggle.py --help
```

**Default naming:**
- Dataset: `nfl-model-v{YYYYMMDD-HHMMSS}`
- Notebook: `nfl-inference-{YYYYMMDD-HHMMSS}`

**What it does:**
1. Verifies Kaggle CLI is installed
2. Checks API key is configured
3. Validates for_kaggle.zip exists
4. Runs model integrity tests
5. Uploads dataset with metadata
6. Creates/updates notebook with dataset attachment
7. Displays upload summary

**Prerequisites:**
- Kaggle CLI: `pip install kaggle`
- API key: `~/.kaggle/kaggle.json`
- for_kaggle.zip exists

---

## Typical Workflows

### Scenario 1: Complete Deployment (Recommended)

```bash
# One command to do everything
bash scripts/full_deployment.sh
```

**What happens:**
1. Builds fresh model with timestamp
2. Uploads to Kaggle
3. Automatically cleans old datasets
4. Shows next testing steps

---

### Scenario 2: Build & Upload Only

```bash
# Build and upload, but keep manual control over cleanup
bash scripts/clean_and_deploy.sh

# Later, manually cleanup:
python scripts/cleanup_old_datasets.py --list-only
python scripts/cleanup_old_datasets.py --delete-old
```

**Use when:**
- You want to verify datasets before cleanup
- You prefer manual API deletions
- You're testing the upload first

---

### Scenario 3: Build Only (Testing)

```bash
# Build and verify locally without uploading
bash scripts/clean_and_deploy.sh --skip-upload

# Or
bash scripts/full_deployment.sh --build-only
```

**Use when:**
- Testing changes without Kaggle upload
- Verifying model pattern in archive
- Development/debugging

---

### Scenario 4: Just Cleanup

```bash
# List datasets to see what needs cleanup
python scripts/cleanup_old_datasets.py --list-only

# Interactively delete old datasets
python scripts/cleanup_old_datasets.py --delete-old
```

**Use when:**
- You uploaded separately
- You need to clean old datasets after testing
- Manual dataset management

---

## Timestamping Strategy

All scripts use consistent timestamp format: `YYYYMMDD-HHMMSS` (or `YYYYMMDD_HHMMSS` for filenames)

**Example:**
- Timestamp: `20251204-101918` (December 4, 2025 at 10:19:18 AM)
- Model file: `nfl_model_v20251204_101918.pkl`
- Dataset name: `nfl-model-v20251204-101918`
- Notebook name: `nfl-inference-20251204-101918`

**Why this matters:**
- Defeats Kaggle aggressive caching
- Each deployment gets unique identifier
- Easy to track which version is deployed
- Audit trail for debugging

---

## Post-Deployment Testing

After running deployment scripts:

### 1. Delete Old Datasets (if not auto-cleaned)

Visit: https://www.kaggle.com/settings/datasets

Delete any datasets matching:
- `nfl-model-v20*` (without time)
- `nfl-model-v20*-*` (with incomplete time)

Keep only the newest with full timestamp.

### 2. Run Notebook on Kaggle

1. Go to: https://www.kaggle.com/code/blazelaserblazer
2. Find newest `nfl-inference-{timestamp}` notebook
3. Click "Commit and Run"

### 3. Verify Output

Look for in the output:
```
✓ LOADING MODEL AND FEATURES
✓ Model Version: 20251204_101918
✓ X model random_state: 42
✓ Y model random_state: 42
✓ NO PCG64BitGenerator errors
✓ predictions.csv created
```

### 4. Submit to Competition

Once verified, enable "Submit to Competition" option in notebook.

---

## Troubleshooting

### "❌ Command failed: kaggle --version"
```bash
# Install Kaggle CLI
pip install kaggle

# Verify installation
kaggle --version
```

### "❌ Kaggle API key not found"
```bash
# 1. Visit: https://www.kaggle.com/settings/account
# 2. Click "Create New Token" (downloads kaggle.json)
# 3. Move file:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/

# 4. Set permissions
chmod 600 ~/.kaggle/kaggle.json

# 5. Verify
ls -la ~/.kaggle/kaggle.json
```

### "❌ Model integrity tests failed"
Check that the model has:
- `random_state=42` (integer, not RandomState object)
- Proper feature scaling and training
- Run: `python -m pytest tests/test_model_integrity.py -v`

### "Archive does NOT contain nfl_model_v*.pkl"
```bash
# Check what's in the archive
unzip -l for_kaggle.zip | grep "\.pkl"

# Clean and rebuild
rm -rf for_kaggle for_kaggle.zip
bash scripts/prepare_for_kaggle.sh
```

### "Notebook loaded wrong model (PCG64BitGenerator error)"
1. This means old dataset is being found first
2. Verify you deleted old datasets from Kaggle UI
3. Check notebook is looking for `nfl_model_v*.pkl` pattern
4. Run notebook again on Kaggle

---

## Script Dependencies

```
full_deployment.sh
├── clean_and_deploy.sh
│   ├── prepare_for_kaggle.sh
│   │   └── tests/test_model_integrity.py
│   └── upload_to_kaggle.py
│       ├── Kaggle CLI
│       └── API key (~/.kaggle/kaggle.json)
└── cleanup_old_datasets.py
    ├── Kaggle CLI
    └── API key (~/.kaggle/kaggle.json)
```

---

## Environment Requirements

All scripts require:
- Python 3.8+
- Virtual environment at `.venv/`
- Kaggle CLI: `pip install kaggle`
- Kaggle API key: `~/.kaggle/kaggle.json`
- Git (for optional --commit flags)

---

## Safety & Best Practices

1. **Always use `--list-only` first** before deleting datasets
2. **Test with `--build-only` first** before full upload
3. **Keep recent backups** of model files locally
4. **Review timestamps** before and after deployment
5. **Check Kaggle UI** to confirm datasets are as expected
6. **Use git** to track version changes

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Failure (check error message) |
| 2 | Invalid arguments |

---

## Example Execution

```bash
# Complete deployment
$ bash scripts/full_deployment.sh

# Output:
# ╔═══════════════════════════════════════════════════════════════════╗
# ║ PHASE 1: Local Build
# ╚═══════════════════════════════════════════════════════════════════╝
# → Step 1/11: Cleaning previous artifacts
# ✓ Cleaned artifacts
# → Step 2/11: Running model integrity tests
# ✓ Model integrity tests passed
# → Step 3/11: Building for_kaggle directory
# ✓ Built for_kaggle with model: nfl_model_v20251204_101918.pkl
# ...
# ✓ DEPLOYMENT COMPLETE!
# 
# Now you need to test on Kaggle:
# 1. Visit: https://www.kaggle.com/code/blazelaserblazer
# 2. Click "Commit and Run" on newest notebook
# 3. Verify output and submit to competition
```

---

## Support

For issues:
1. Check script logs for error messages
2. Verify Kaggle API key and CLI installation
3. Test components individually:
   - `bash scripts/prepare_for_kaggle.sh`
   - `python scripts/upload_to_kaggle.py`
   - `python scripts/cleanup_old_datasets.py --list-only`
4. Review model integrity tests: `python -m pytest tests/test_model_integrity.py -v`
