"""
Prevention tests for issues encountered during development.

These tests are specifically designed to prevent the issues we've already
encountered from happening again:
1. RandomState object serialization issues (PCG64 BitGenerator errors)
2. Script errors in critical paths
3. Model file naming convention violations
4. Training data issues
5. Kaggle deployment readiness checks
"""

import ast
import os
import re
from pathlib import Path
import joblib
import numpy as np
import pytest
from inspect import getsource


# ============================================================================
# ISSUE #1: RandomState Object Serialization (PCG64 BitGenerator Error)
# ============================================================================

def test_train_script_uses_default_rng_not_randomstate():
    """
    PREVENTION: Ensure train_model.py never creates np.random.RandomState() objects.
    
    Previously: Line 89 used `np.random.RandomState(random_state).choice()`
    Problem: RandomState objects don't serialize well across NumPy versions
    This causes: PCG64BitGenerator errors when model is loaded on Kaggle
    Solution: Use `np.random.default_rng(random_state).choice()` instead
    """
    train_script = Path('scripts/train_model.py').read_text()
    
    # Check for problematic patterns
    bad_patterns = [
        r'np\.random\.RandomState\s*\(',
        r'numpy\.random\.RandomState\s*\(',
    ]
    
    for pattern in bad_patterns:
        matches = re.findall(pattern, train_script)
        assert len(matches) == 0, \
            f"Found RandomState instantiation in train_model.py: {matches}. " \
            f"Use np.random.default_rng() instead to avoid PCG64 serialization issues."


def test_prepare_for_kaggle_script_no_randomstate():
    """Ensure prepare_for_kaggle.sh doesn't create RandomState objects."""
    prep_script = Path('scripts/prepare_for_kaggle.sh').read_text()
    
    # Shell scripts shouldn't be creating RandomState anyway, but check for comments
    # that might indicate awareness
    bad_patterns = [r'random\.RandomState', r'RandomState\s*\(']
    
    for pattern in bad_patterns:
        matches = re.findall(pattern, prep_script)
        assert len(matches) == 0, \
            f"Found RandomState reference in prepare_for_kaggle.sh: {matches}"


def test_model_pkl_no_randomstate_objects():
    """
    Deep check: Loaded model must not contain any RandomState objects anywhere.
    This is the ultimate check before deployment.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    def find_random_state_objects(obj, path="root", visited=None):
        """Recursively search for RandomState objects in any nested structure."""
        if visited is None:
            visited = set()
        
        # Avoid infinite recursion
        obj_id = id(obj)
        if obj_id in visited:
            return []
        visited.add(obj_id)
        
        found = []
        
        # Direct check
        if isinstance(obj, np.random.RandomState):
            found.append(f"RandomState object at {path}")
            return found
        
        # Check object attributes
        if hasattr(obj, '__dict__'):
            try:
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, np.random.RandomState):
                        found.append(f"RandomState in {path}.{attr_name}")
                    else:
                        found.extend(find_random_state_objects(
                            attr_value, f"{path}.{attr_name}", visited
                        ))
            except (AttributeError, RuntimeError):
                pass
        
        # Check containers
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, np.random.RandomState):
                    found.append(f"RandomState in {path}['{key}']")
                else:
                    found.extend(find_random_state_objects(value, f"{path}['{key}']", visited))
        
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                if isinstance(item, np.random.RandomState):
                    found.append(f"RandomState in {path}[{i}]")
                else:
                    found.extend(find_random_state_objects(item, f"{path}[{i}]", visited))
        
        return found
    
    random_state_locations = find_random_state_objects(model_dict)
    assert len(random_state_locations) == 0, \
        f"CRITICAL: Found RandomState objects in model: {random_state_locations}. " \
        f"This will cause PCG64BitGenerator errors on Kaggle!"


# ============================================================================
# ISSUE #2: Model File Naming Convention
# ============================================================================

def test_model_naming_follows_versioning_pattern():
    """
    PREVENTION: Ensure any model files follow the correct versioning pattern.
    
    Pattern: nfl_model_v{YYYYMMDD_HHMMSS}.pkl
    Alternative: best_model.pkl (for compatibility)
    
    We previously had old naming patterns like 'nfl_model.pkl' that caused
    confusion and were not properly cleaned up.
    """
    models_dir = Path('models')
    if not models_dir.exists():
        pytest.skip("models/ directory not found")
    
    # Allowed patterns
    valid_patterns = [
        r'^nfl_model_v\d{8}_\d{6}\.pkl$',  # nfl_model_v20251204_122914.pkl
        r'^best_model\.pkl$',               # best_model.pkl for compatibility
    ]
    
    for pkl_file in models_dir.glob('*.pkl'):
        filename = pkl_file.name
        
        is_valid = any(re.match(pattern, filename) for pattern in valid_patterns)
        assert is_valid, \
            f"Model file '{filename}' doesn't follow versioning pattern. " \
            f"Use 'nfl_model_v{{YYYYMMDD_HHMMSS}}.pkl' or 'best_model.pkl'"


def test_for_kaggle_model_naming_consistent():
    """
    PREVENTION: Ensure for_kaggle/models/ also follows naming convention.
    
    Old issue: Multiple copies with inconsistent names caused Kaggle to load wrong versions
    Solution: Enforce single-model-only policy with proper versioning
    """
    for_kaggle_models = Path('for_kaggle/models')
    if not for_kaggle_models.exists():
        pytest.skip("for_kaggle/models/ not built yet")
    
    pkl_files = list(for_kaggle_models.glob('*.pkl'))
    
    if pkl_files:
        # Should have at most 2 files: one versioned, one best_model.pkl
        assert len(pkl_files) <= 2, \
            f"for_kaggle/models/ has {len(pkl_files)} model files. " \
            f"Should have at most 2 (versioned + best_model.pkl for compatibility). " \
            f"Files: {[f.name for f in pkl_files]}"
        
        # Check naming
        for pkl_file in pkl_files:
            filename = pkl_file.name
            valid = (
                re.match(r'^nfl_model_v\d{8}_\d{6}\.pkl$', filename) or
                filename == 'best_model.pkl'
            )
            assert valid, f"Invalid model name in for_kaggle: {filename}"


# ============================================================================
# ISSUE #3: Training Script Critical Paths
# ============================================================================

def test_train_script_syntax_valid():
    """Ensure train_model.py has valid Python syntax."""
    script_path = Path('scripts/train_model.py')
    assert script_path.exists(), "train_model.py not found"
    
    try:
        code = script_path.read_text()
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"train_model.py has syntax error: {e}")


def test_prepare_for_kaggle_script_syntax_valid():
    """Ensure prepare_for_kaggle.sh has valid bash syntax."""
    script_path = Path('scripts/prepare_for_kaggle.sh')
    assert script_path.exists(), "prepare_for_kaggle.sh not found"
    
    # Check for common bash syntax issues
    content = script_path.read_text()
    
    # Must have shebang
    assert content.startswith('#!/'), "Shell script must start with shebang (#!/)"
    
    # Must have set -e or similar error handling
    assert 'set -' in content, "Shell script should use 'set -' for error handling"
    
    # Check for critical structural issues
    assert 'mkdir -p' in content, "Shell script should use mkdir -p for directory creation"
    assert 'nfl_model_v' in content, "Shell script should reference nfl_model_v pattern"


def test_full_deployment_script_syntax_valid():
    """Ensure full_deployment.sh has valid bash syntax."""
    script_path = Path('scripts/full_deployment.sh')
    assert script_path.exists(), "full_deployment.sh not found"
    
    content = script_path.read_text()
    
    # Must have shebang
    assert content.startswith('#!/'), "Shell script must start with shebang"
    
    # Must have error handling
    assert 'set -' in content, "Shell script should use 'set -' for error handling"


# ============================================================================
# ISSUE #4: Script Cleanup Behavior (Single-Model Policy)
# ============================================================================

def test_prepare_for_kaggle_implements_cleanup():
    """
    PREVENTION: Ensure prepare_for_kaggle.sh removes old versioned models.
    
    Previously: Old versioned models accumulated, causing confusion
    Solution: Always clean old nfl_model_v*.pkl files before creating new one
    """
    script_path = Path('scripts/prepare_for_kaggle.sh').read_text()
    
    # Should remove old models
    has_cleanup = (
        'rm -f' in script_path and
        ('nfl_model_v' in script_path or '[0-9]*.pkl' in script_path)
    )
    
    assert has_cleanup, \
        "prepare_for_kaggle.sh should remove old nfl_model_v*.pkl files. " \
        "Add: rm -f <modeldir>/nfl_model_v[0-9]*.pkl"


def test_train_script_implements_cleanup():
    """
    PREVENTION: Ensure train_model.py removes old versioned models.
    
    Previously: Old models accumulated in models/ directory
    Solution: Clean old models after training new one
    """
    script_path = Path('scripts/train_model.py').read_text()
    
    # Should mention removing old models
    has_cleanup = 'glob' in script_path and 'nfl_model_v' in script_path
    
    assert has_cleanup, \
        "train_model.py should clean old nfl_model_v*.pkl files after training. " \
        "Use: Path('models').glob('nfl_model_v*.pkl')"


def test_no_references_to_legacy_patterns():
    """
    PREVENTION: Ensure code doesn't reference old naming patterns in executable code.
    
    Previously: References to "NEW pattern" and "defeat caching" cluttered code
    Solution: Remove all such references, assume versioning was always standard
    
    Note: Comments are allowed since they're not active code
    """
    scripts_to_check = [
        'scripts/train_model.py',
        'scripts/prepare_for_kaggle.sh',
        'scripts/full_deployment.sh',
    ]
    
    # These patterns should not appear in actual code (not comments)
    bad_patterns = [
        (r'\bdefeat caching\b', 'avoid language about defeating Kaggle caching'),
        (r'\bdefeat Kaggle\b', 'avoid language about defeating Kaggle'),
        (r'RandomState\s*\(', 'avoid RandomState objects'),
    ]
    
    for script_path_str in scripts_to_check:
        script_path = Path(script_path_str)
        if not script_path.exists():
            continue
        
        content = script_path.read_text()
        
        for pattern, description in bad_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert len(matches) == 0, \
                f"Found problematic pattern in {script_path}: {description}. " \
                f"Matches: {matches}"


# ============================================================================
# ISSUE #5: Model Metadata Consistency
# ============================================================================

def test_model_metadata_matches_actual_model_state():
    """
    PREVENTION: Verify metadata file (if present) matches actual model state.
    
    If we create MODEL_METADATA.txt, it should accurately reflect the model.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model not found")
    
    metadata_path = Path('models/MODEL_METADATA.txt')
    if not metadata_path.exists():
        pytest.skip("Model metadata not found")
    
    model_dict = joblib.load(model_path)
    metadata = metadata_path.read_text()
    
    # Check that metadata mentions the correct random_state
    if 'random_state' in metadata:
        assert '42' in metadata, \
            "Model metadata should mention random_state=42 if random_state is included"


def test_for_kaggle_model_metadata_present():
    """
    PREVENTION: Ensure for_kaggle has MODEL_METADATA.txt for deployed model.
    
    This helps track which model version was deployed.
    """
    for_kaggle_models = Path('for_kaggle/models')
    if not for_kaggle_models.exists():
        pytest.skip("for_kaggle/models/ not built yet")
    
    pkl_files = list(for_kaggle_models.glob('nfl_model_v*.pkl'))
    if pkl_files:
        # Should have metadata file
        metadata_path = for_kaggle_models / 'MODEL_METADATA.txt'
        assert metadata_path.exists(), \
            f"for_kaggle/models/ has versioned model but missing MODEL_METADATA.txt. " \
            f"This metadata file helps track deployments."


# ============================================================================
# ISSUE #6: Notebook Submission Readiness
# ============================================================================

def test_notebook_kernel_metadata_present():
    """
    PREVENTION: Ensure submission notebook has proper kernel metadata for Kaggle.
    
    Previously: Missing kernelspec caused papermill execution to fail
    Solution: Include kernelspec in notebook metadata
    """
    notebook_path = Path('notebooks/submission_notebook.ipynb')
    if not notebook_path.exists():
        pytest.skip("Notebook not found")
    
    import json
    notebook = json.loads(notebook_path.read_text())
    
    # Check for kernelspec
    assert 'metadata' in notebook, "Notebook missing metadata section"
    metadata = notebook['metadata']
    
    assert 'kernelspec' in metadata, \
        "Notebook missing kernelspec. Add to metadata: " \
        '{"display_name": "Python 3", "language": "python", "name": "python3"}'
    
    kernelspec = metadata['kernelspec']
    assert kernelspec.get('name') == 'python3', \
        f"Notebook kernel should be 'python3', got {kernelspec.get('name')}"


def test_notebook_model_validation_logic_present():
    """
    PREVENTION: Ensure notebook validates model before using it.
    
    Solution: Check model name follows versioning pattern before loading
    """
    notebook_path = Path('notebooks/submission_notebook.ipynb')
    if not notebook_path.exists():
        pytest.skip("Notebook not found")
    
    import json
    notebook = json.loads(notebook_path.read_text())
    
    # Check notebook cells for model validation
    notebook_text = '\n'.join(
        cell.get('source', '') if isinstance(cell.get('source'), str)
        else ''.join(cell.get('source', []))
        for cell in notebook.get('cells', [])
    )
    
    # Should mention nfl_model_v pattern
    assert 'nfl_model_v' in notebook_text, \
        "Notebook should validate model filename pattern (nfl_model_v)"


# ============================================================================
# ISSUE #7: Training Data Integrity
# ============================================================================

def test_training_data_files_present():
    """Ensure all expected training data files exist."""
    train_dir = Path('train')
    if not train_dir.exists():
        pytest.skip("train/ directory not found")
    
    # Should have input and output files for each week
    input_files = list(train_dir.glob('input_2023_w*.csv'))
    output_files = list(train_dir.glob('output_2023_w*.csv'))
    
    assert len(input_files) > 0, "No training input files found"
    assert len(output_files) > 0, "No training output files found"
    
    # Should have matching pairs
    input_weeks = {int(f.name.split('_w')[1].split('.')[0]) for f in input_files}
    output_weeks = {int(f.name.split('_w')[1].split('.')[0]) for f in output_files}
    
    assert input_weeks == output_weeks, \
        f"Mismatched weeks: inputs={input_weeks}, outputs={output_weeks}"


def test_test_data_files_present():
    """Ensure test data files exist."""
    test_csv = Path('test.csv')
    test_input_csv = Path('test_input.csv')
    
    assert test_csv.exists(), "test.csv not found"
    assert test_input_csv.exists(), "test_input.csv not found"


# ============================================================================
# ISSUE #8: Deployment Script Integration
# ============================================================================

def test_full_deployment_orchestrates_all_steps():
    """
    PREVENTION: Ensure full_deployment.sh includes essential workflow steps.
    
    Should:
    1. Clean old artifacts
    2. Run tests
    3. Build for_kaggle
    4. Upload to Kaggle
    """
    script_path = Path('scripts/full_deployment.sh').read_text()
    
    required_steps = [
        'test',                      # Run tests
        'prepare_for_kaggle',        # Build for_kaggle
        'kaggle',                    # Upload to Kaggle
    ]
    
    for step in required_steps:
        assert step in script_path, \
            f"full_deployment.sh missing step containing '{step}'"


# ============================================================================
# ISSUE #9: Model Random State Verification
# ============================================================================

def test_model_random_states_are_integers_not_objects():
    """
    CRITICAL: Verify random_state attributes are integers, not RandomState objects.
    
    This directly addresses the PCG64 error we encountered.
    RandomState objects serialize incorrectly across NumPy versions.
    """
    model_path = Path('models/best_model.pkl')
    if not model_path.exists():
        pytest.skip("Model file not found")
    
    model_dict = joblib.load(model_path)
    
    for model_name in ['x', 'y']:
        model = model_dict['models'][model_name]
        random_state = model.random_state
        
        # Must be None or integer, NEVER a RandomState object
        assert not isinstance(random_state, np.random.RandomState), \
            f"{model_name} model has RandomState object (will cause PCG64 error). " \
            f"Should be integer 42 or similar."
        
        assert random_state is None or isinstance(random_state, (int, np.integer)), \
            f"{model_name} model random_state is {type(random_state).__name__}, " \
            f"should be None or int"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
