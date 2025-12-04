# Test Suite

This directory contains unit tests to ensure model integrity and submission readiness for the NFL Big Data Bowl 2026 competition.

## Test Files

### `test_model_integrity.py`

**Critical validation tests** that verify model file compatibility and structure. These tests would have caught the NumPy PCG64 BitGenerator compatibility issue that occurred when transferring models between environments with different NumPy versions.

**Key Tests:**

1. **`test_model_random_state_compatibility`** ⭐ 
   - **Most Important**: Verifies that `random_state` attributes are integers or None, NOT RandomState objects
   - Prevents the NumPy PCG64 BitGenerator error that breaks model loading across different NumPy versions
   - This test would have caught the original issue before uploading to Kaggle

2. **`test_model_no_numpy_randomstate_objects`** ⭐
   - Deep inspection of entire model tree to find hidden RandomState objects
   - Recursively checks all model attributes to ensure cross-version compatibility
   - Extra safety layer beyond basic random_state check

3. **`test_model_loads_successfully`**
   - Basic smoke test: can the model file be loaded without errors?
   - Catches corrupted pickle files or missing dependencies

4. **`test_model_structure`**
   - Verifies expected dictionary structure with `models`, `feature_columns`, etc.
   - Ensures both `x` and `y` coordinate models are present

5. **`test_model_can_be_pickled_and_unpickled`**
   - Simulates the serialization cycle when uploading to Kaggle
   - Catches any objects that can't survive pickle/unpickle

6. **`test_model_makes_valid_predictions`**
   - End-to-end test with dummy data
   - Verifies predictions are finite numbers (no NaN or inf)
   - Confirms both models can actually generate predictions

7. **`test_model_feature_columns_valid`**
   - Validates feature metadata is properly structured
   - Ensures feature names are strings and list is non-empty

### `test_submission_ready.py`

Integration tests for the full submission pipeline including feature engineering and data loading.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only model integrity tests
pytest tests/test_model_integrity.py -v

# Run a specific test
pytest tests/test_model_integrity.py::test_model_random_state_compatibility -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## When to Run These Tests

- **Before training**: Set up test infrastructure
- **After training**: Immediately verify model integrity before packaging
- **Before uploading to Kaggle**: Final validation to catch compatibility issues (automatically runs when using `make prepare-for-kaggle`)
- **In CI/CD**: Automated validation on every commit
- **After fixing models**: Confirm fixes worked as expected

## Automated Integration

The model integrity tests are **automatically run** by the `prepare_for_kaggle.sh` script. When you run:

```bash
make prepare-for-kaggle
# or
./scripts/prepare_for_kaggle.sh --zip
```

The script will:
1. Run all model integrity tests
2. **Stop with an error** if any test fails (exit code 1)
3. Only proceed to package files if all tests pass

This ensures you can never accidentally package and upload a broken model to Kaggle!

## What the Tests Caught

The model integrity tests were created after discovering that scikit-learn models with `random_state=np.random.RandomState(42)` failed to load on Kaggle with this error:

```
GatewayRuntimeError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module
```

This happened because:
1. Models were trained with one version of NumPy
2. Kaggle's inference environment used a different NumPy version
3. RandomState objects contain version-specific state that can't transfer

The solution was to use `random_state=42` (an integer) instead of `random_state=np.random.RandomState(42)`.

**These tests would have caught this issue immediately**, before spending time debugging on Kaggle.

## Test Philosophy

These tests prioritize:
- **Prevention**: Catch issues before deployment
- **Speed**: Fast enough to run frequently (< 2 seconds)
- **Clarity**: Clear error messages explaining what went wrong
- **Practicality**: Focus on real issues we've encountered

## Future Enhancements

Consider adding:
- Tests for feature engineering pipeline correctness
- Validation of prediction ranges (x: -10 to 130, y: -10 to 100)
- Performance benchmarks
- Memory usage checks for large batch predictions
- Integration tests with actual Kaggle submission format
