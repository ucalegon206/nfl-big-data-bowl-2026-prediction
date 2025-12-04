# Test Suite Quick Reference

## Running Tests

```bash
# Everything
pytest tests/ -v

# Just prevention tests
pytest tests/test_prevention_future_issues.py -v

# Just model integrity tests
pytest tests/test_model_integrity.py -v

# Just submission ready tests
pytest tests/test_submission_ready.py -v

# Specific test
pytest tests/test_prevention_future_issues.py::test_model_pkl_no_randomstate_objects -v
```

## What Each Test Category Catches

### Model Integrity Tests (12 tests - `test_model_integrity.py`)
Validates the trained model file itself:
- ✅ Model file exists and loads
- ✅ Model structure (dict with 'models' and 'feature_columns' keys)
- ✅ No RandomState objects (PCG64 compatibility)
- ✅ Can pickle/unpickle successfully
- ✅ Feature columns are valid
- ✅ Makes valid predictions
- ✅ Has reasonable metadata
- ✅ Random state is exactly 42
- ✅ File size is reasonable
- ✅ NumPy version info for debugging

### Prevention Tests (19 tests - `test_prevention_future_issues.py`)
Prevents issues from happening again:

**RandomState Issues (3 tests)**
- Training script doesn't use RandomState()
- Loaded model has no RandomState objects
- Model random_state values are integers

**Naming Convention (2 tests)**
- models/ directory follows nfl_model_v{YYYYMMDD_HHMMSS}.pkl pattern
- for_kaggle/models/ follows same pattern

**Script Quality (3 tests)**
- train_model.py valid Python syntax
- prepare_for_kaggle.sh valid bash syntax
- full_deployment.sh valid bash syntax

**Cleanup Implementation (2 tests)**
- prepare_for_kaggle.sh removes old models
- train_model.py removes old models

**Legacy Patterns (1 test)**
- No "defeat caching", "NEW pattern" references

**Metadata (2 tests)**
- MODEL_METADATA.txt matches model state
- for_kaggle has metadata file

**Notebook (2 tests)**
- Notebook has kernelspec metadata
- Notebook validates model filename pattern

**Data (2 tests)**
- Training data files present with matching weeks
- Test data files present

**Deployment (1 test)**
- full_deployment.sh includes required steps

### Submission Ready Tests (2 tests - `test_submission_ready.py`)
Feature pipeline and quick prediction check:
- ✅ Test input loads and features process correctly
- ✅ Model metadata valid and predictions work (if model exists)

## Failure Interpretation Guide

| Test Fails | Likely Cause | Fix |
|-----------|-------------|-----|
| `test_model_pkl_no_randomstate_objects` | Old model with RandomState | Retrain with `scripts/train_model.py` |
| `test_model_naming_follows_versioning_pattern` | Invalid model filename | Delete and retrain, follow pattern |
| `test_train_script_uses_default_rng_not_randomstate` | Code has RandomState() | Check line 89 in train_model.py |
| `test_prepare_for_kaggle_implements_cleanup` | No cleanup logic | Add `rm -f` commands |
| `test_notebook_kernel_metadata_present` | Missing kernelspec | Add to notebook metadata |
| `test_training_data_files_present` | Data missing | Download from Kaggle |
| `test_test_data_files_present` | Test data missing | Download from Kaggle |

## When Tests Run Automatically

1. **Before preparing for Kaggle**: `prepare_for_kaggle.sh` runs model integrity tests
2. **Before full deployment**: `full_deployment.sh` runs all tests
3. **CI/CD**: Tests can run on every commit to catch regressions

## Adding New Tests

1. Open `tests/test_prevention_future_issues.py`
2. Add new test function with descriptive name and docstring
3. Run: `pytest tests/test_prevention_future_issues.py::your_new_test -v`
4. Commit: `git add tests/test_prevention_future_issues.py && git commit -m "Add test for X"`

## Understanding Test Organization

```
tests/
├── __init__.py
├── test_model_integrity.py          # 12 tests: Model file validation
├── test_prevention_future_issues.py # 19 tests: Prevent known problems
└── test_submission_ready.py          # 2 tests: Features and predictions
```

Total: **33 tests**, ~1 second execution time, all critical paths covered.
