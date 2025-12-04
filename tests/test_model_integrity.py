"""
Unit tests to validate model file integrity and catch compatibility issues.

These tests specifically check for:
1. NumPy random state compatibility (PCG64 BitGenerator issues)
2. Model structure and required keys
3. Serialization/deserialization integrity
4. Feature compatibility
"""
import joblib
import numpy as np
import pytest
from pathlib import Path
import pickle
import sys


def test_model_file_exists():
    """Verify the model file exists at the expected location."""
    model_path = Path('models/best_model.pkl')
    assert model_path.exists(), f"Model file not found at {model_path}"


def test_model_loads_successfully():
    """Verify the model can be loaded without errors."""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    try:
        model_dict = joblib.load(model_path)
        assert model_dict is not None, "Model loaded as None"
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


def test_model_structure():
    """Verify the model has the expected dictionary structure."""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    # Check it's a dictionary
    assert isinstance(model_dict, dict), f"Expected dict, got {type(model_dict)}"
    
    # Check required keys
    required_keys = ['models', 'feature_columns']
    for key in required_keys:
        assert key in model_dict, f"Missing required key: {key}"
    
    # Check models sub-dictionary
    assert isinstance(model_dict['models'], dict), "models should be a dict"
    assert 'x' in model_dict['models'], "Missing 'x' model"
    assert 'y' in model_dict['models'], "Missing 'y' model"


def test_model_random_state_compatibility():
    """
    CRITICAL TEST: Verify random_state is serialization-compatible.
    
    This test would have caught the NumPy PCG64 BitGenerator issue.
    Random states must be either:
    - None
    - An integer
    - NOT a RandomState object (which causes cross-version compatibility issues)
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    for model_name in ['x', 'y']:
        model = model_dict['models'][model_name]
        
        # Check if model has random_state attribute
        if hasattr(model, 'random_state'):
            random_state = model.random_state
            
            # Random state should be None or an integer, NOT a RandomState object
            assert random_state is None or isinstance(random_state, (int, np.integer)), \
                f"{model_name} model has incompatible random_state type: {type(random_state)}. " \
                f"Must be None or int, not RandomState object. " \
                f"This causes NumPy PCG64 BitGenerator errors across different NumPy versions."
            
            if isinstance(random_state, (int, np.integer)):
                # Verify it's a reasonable seed value
                assert 0 <= random_state < 2**32, \
                    f"{model_name} model random_state {random_state} out of valid range"


def test_model_no_numpy_randomstate_objects():
    """
    Deep inspection: Ensure no RandomState objects anywhere in model tree.
    
    This is a more thorough check that traverses the entire model structure
    looking for problematic RandomState objects that could cause PCG64 errors.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    def check_for_random_state_objects(obj, path="root"):
        """Recursively check for RandomState objects."""
        # Check if this object itself is a RandomState
        if isinstance(obj, np.random.RandomState):
            return False, f"Found RandomState object at {path}"
        
        # Check attributes if it's a custom object
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, np.random.RandomState):
                    return False, f"Found RandomState object at {path}.{attr_name}"
                # Recursively check nested objects
                if hasattr(attr_value, '__dict__') or isinstance(attr_value, (list, tuple, dict)):
                    ok, msg = check_for_random_state_objects(attr_value, f"{path}.{attr_name}")
                    if not ok:
                        return False, msg
        
        # Check list/tuple elements
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                if isinstance(item, np.random.RandomState):
                    return False, f"Found RandomState object at {path}[{i}]"
        
        # Check dict values
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, np.random.RandomState):
                    return False, f"Found RandomState object at {path}['{key}']"
        
        return True, "OK"
    
    # Check both models
    for model_name in ['x', 'y']:
        model = model_dict['models'][model_name]
        ok, msg = check_for_random_state_objects(model, f"models['{model_name}']")
        assert ok, f"CRITICAL: {msg}. This will cause NumPy PCG64 compatibility errors!"


def test_model_can_be_pickled_and_unpickled():
    """
    Verify the model can survive a pickle/unpickle cycle.
    
    This simulates what happens when uploading to Kaggle or transferring
    between environments with different NumPy versions.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    # Load original
    original = joblib.load(model_path)
    
    # Pickle and unpickle
    try:
        pickled = pickle.dumps(original)
        unpickled = pickle.loads(pickled)
        
        # Verify structure preserved
        assert isinstance(unpickled, dict)
        assert 'models' in unpickled
        assert 'x' in unpickled['models']
        assert 'y' in unpickled['models']
        
    except Exception as e:
        pytest.fail(f"Failed pickle/unpickle cycle: {e}")


def test_model_feature_columns_valid():
    """Verify feature_columns is a valid list of strings."""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    assert 'feature_columns' in model_dict
    feature_columns = model_dict['feature_columns']
    
    assert isinstance(feature_columns, list), "feature_columns must be a list"
    assert len(feature_columns) > 0, "feature_columns cannot be empty"
    
    for col in feature_columns:
        assert isinstance(col, str), f"Feature column must be string, got {type(col)}: {col}"


def test_model_makes_valid_predictions():
    """Verify models can make predictions on dummy data."""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    # Create dummy input data with correct number of features
    n_features = len(model_dict['feature_columns'])
    X_dummy = np.random.randn(10, n_features)
    
    # Test x model
    x_model = model_dict['models']['x']
    x_pred = x_model.predict(X_dummy)
    assert x_pred.shape == (10,), f"Expected shape (10,), got {x_pred.shape}"
    assert not np.isnan(x_pred).any(), "X predictions contain NaN"
    assert np.isfinite(x_pred).all(), "X predictions contain inf"
    
    # Test y model
    y_model = model_dict['models']['y']
    y_pred = y_model.predict(X_dummy)
    assert y_pred.shape == (10,), f"Expected shape (10,), got {y_pred.shape}"
    assert not np.isnan(y_pred).any(), "Y predictions contain NaN"
    assert np.isfinite(y_pred).all(), "Y predictions contain inf"


def test_model_has_reasonable_metadata():
    """Verify model has expected metadata fields."""
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    # Check for expected metadata
    assert 'feature_columns' in model_dict, "Missing feature_columns metadata"
    assert len(model_dict['feature_columns']) > 0, "Empty feature_columns"
    
    # Check for player_position_values if present
    if 'player_position_values' in model_dict:
        assert isinstance(model_dict['player_position_values'], (dict, list)), \
            "player_position_values should be a dict or list"


def test_model_random_state_is_42():
    """
    Verify that both models have random_state set to exactly 42.
    
    This is our standard seed value after fixing the NumPy compatibility issue.
    If this test fails, the model may not have been properly fixed.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    # Check both models have random_state=42
    x_random_state = model_dict['models']['x'].random_state
    y_random_state = model_dict['models']['y'].random_state
    
    assert x_random_state == 42, f"X model random_state should be 42, got {x_random_state}"
    assert y_random_state == 42, f"Y model random_state should be 42, got {y_random_state}"
    
    print(f"\n✓ Both models have random_state=42")


def test_model_file_size_reasonable():
    """
    Verify model file size is within reasonable bounds.
    
    A model that's too small might be corrupted or empty.
    A model that's too large might have issues or extra data.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    file_size = model_path.stat().st_size
    
    # Model should be between 100KB and 100MB
    assert file_size > 100_000, f"Model file too small: {file_size} bytes (might be corrupted)"
    assert file_size < 100_000_000, f"Model file too large: {file_size} bytes (might have issues)"
    
    print(f"\n✓ Model file size is reasonable: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")


def test_numpy_version_info():
    """
    Print NumPy version info for debugging.
    
    This helps identify which NumPy version the tests are running with,
    useful for diagnosing compatibility issues.
    """
    print(f"\nNumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    
    # This test always passes, it just prints info
    assert True


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
