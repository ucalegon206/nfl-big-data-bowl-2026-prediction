"""
Deployment Safety Tests for NFL Big Data Bowl 2026

These tests verify that our models and deployment process avoid common issues:
1. Pickle serialization issues (PCG64BitGenerator, RandomState)
2. Model file format compatibility
3. Kaggle environment compatibility
4. Ensemble integrity and loading
"""

import pytest
import json
from pathlib import Path
import sys
import tempfile
import shutil

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestModelSerializationSafety:
    """Test that models don't use pickle or problematic serialization."""
    
    def test_ensemble_directory_exists(self):
        """Check that an ensemble directory exists."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = list(models_dir.glob('nfl_ensemble_v*'))
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found - run train_ensemble.py first")
        
        assert len(ensemble_dirs) >= 1, "At least one ensemble directory should exist"
    
    def test_ensemble_uses_native_formats(self):
        """Verify ensemble uses LightGBM .txt and XGBoost .json formats."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        
        # Check for LightGBM .txt files
        lgb_files = list(ensemble_dir.glob('lgb_model_*.txt'))
        assert len(lgb_files) == 2, f"Expected 2 LightGBM .txt files, found {len(lgb_files)}"
        
        # Check for XGBoost .json files
        xgb_files = list(ensemble_dir.glob('xgb_model_*.json'))
        assert len(xgb_files) == 2, f"Expected 2 XGBoost .json files, found {len(xgb_files)}"
    
    def test_ensemble_has_no_pickle_files(self):
        """Ensure ensemble directory contains no .pkl files."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        pkl_files = list(ensemble_dir.glob('*.pkl'))
        
        assert len(pkl_files) == 0, f"Found pickle files in ensemble: {[f.name for f in pkl_files]}"
    
    def test_metadata_is_valid_json(self):
        """Verify metadata.json is valid JSON without Python objects."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        metadata_path = ensemble_dir / 'metadata.json'
        
        assert metadata_path.exists(), "metadata.json not found"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check required keys
        required_keys = ['feature_columns', 'ensemble_weights', 'model_files']
        for key in required_keys:
            assert key in metadata, f"Missing required key: {key}"
        
        # Verify feature_columns is a list
        assert isinstance(metadata['feature_columns'], list)
        assert len(metadata['feature_columns']) > 0
        
        # Verify ensemble_weights
        weights = metadata['ensemble_weights']
        assert 'lightgbm' in weights
        assert 'xgboost' in weights
        assert abs(weights['lightgbm'] + weights['xgboost'] - 1.0) < 0.01  # Should sum to ~1.0
    
    def test_lightgbm_models_loadable(self):
        """Test that LightGBM models can be loaded using native API."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not installed")
        
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        
        lgb_x_path = ensemble_dir / 'lgb_model_x.txt'
        lgb_y_path = ensemble_dir / 'lgb_model_y.txt'
        
        assert lgb_x_path.exists(), "lgb_model_x.txt not found"
        assert lgb_y_path.exists(), "lgb_model_y.txt not found"
        
        # Load models
        lgb_x = lgb.Booster(model_file=str(lgb_x_path))
        lgb_y = lgb.Booster(model_file=str(lgb_y_path))
        
        # Verify models are loaded
        assert lgb_x.num_trees() > 0
        assert lgb_y.num_trees() > 0
    
    def test_xgboost_models_loadable(self):
        """Test that XGBoost models can be loaded using native API."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost not installed")
        
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        
        xgb_x_path = ensemble_dir / 'xgb_model_x.json'
        xgb_y_path = ensemble_dir / 'xgb_model_y.json'
        
        assert xgb_x_path.exists(), "xgb_model_x.json not found"
        assert xgb_y_path.exists(), "xgb_model_y.json not found"
        
        # Load models
        xgb_x = xgb.Booster()
        xgb_x.load_model(str(xgb_x_path))
        
        xgb_y = xgb.Booster()
        xgb_y.load_model(str(xgb_y_path))
        
        # Verify models are loaded
        assert xgb_x.num_boosted_rounds() > 0
        assert xgb_y.num_boosted_rounds() > 0


class TestEnsembleIntegrity:
    """Test ensemble structure and consistency."""
    
    def test_all_model_files_exist(self):
        """Verify all model files listed in metadata exist."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        metadata_path = ensemble_dir / 'metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_files = metadata['model_files']
        for model_type, filename in model_files.items():
            model_path = ensemble_dir / filename
            assert model_path.exists(), f"Model file missing: {filename}"
    
    def test_features_consistency(self):
        """Ensure feature list is consistent across metadata."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        metadata_path = ensemble_dir / 'metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feat_cols = metadata['feature_columns']
        
        # Check for duplicate features
        assert len(feat_cols) == len(set(feat_cols)), "Duplicate features found"
        
        # Check that features are strings
        assert all(isinstance(f, str) for f in feat_cols), "All features should be strings"
        
        # Check minimum number of features
        assert len(feat_cols) >= 5, f"Too few features: {len(feat_cols)}"
    
    def test_ensemble_weights_valid(self):
        """Verify ensemble weights are reasonable."""
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        metadata_path = ensemble_dir / 'metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        weights = metadata['ensemble_weights']
        
        # Weights should be positive
        assert weights['lightgbm'] > 0, "LightGBM weight must be positive"
        assert weights['xgboost'] > 0, "XGBoost weight must be positive"
        
        # Weights should sum to approximately 1.0
        total = weights['lightgbm'] + weights['xgboost']
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"


class TestNotebookCompatibility:
    """Test that notebook can load and use the ensemble."""
    
    def test_notebook_imports_succeed(self):
        """Verify all notebook imports work."""
        try:
            import pandas as pd
            import numpy as np
            import lightgbm as lgb
            import xgboost as xgb
            import polars as pl
        except ImportError as e:
            pytest.fail(f"Notebook import failed: {e}")
    
    def test_ensemble_prediction_runs(self):
        """Test that ensemble can make predictions on dummy data."""
        try:
            import lightgbm as lgb
            import xgboost as xgb
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("Required packages not installed")
        
        models_dir = REPO_ROOT / 'models'
        ensemble_dirs = sorted(models_dir.glob('nfl_ensemble_v*'), reverse=True)
        
        if not ensemble_dirs:
            pytest.skip("No ensemble directory found")
        
        ensemble_dir = ensemble_dirs[0]
        metadata_path = ensemble_dir / 'metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load models
        lgb_x = lgb.Booster(model_file=str(ensemble_dir / metadata['model_files']['lgb_x']))
        lgb_y = lgb.Booster(model_file=str(ensemble_dir / metadata['model_files']['lgb_y']))
        
        xgb_x = xgb.Booster()
        xgb_x.load_model(str(ensemble_dir / metadata['model_files']['xgb_x']))
        xgb_y = xgb.Booster()
        xgb_y.load_model(str(ensemble_dir / metadata['model_files']['xgb_y']))
        
        # Create dummy data
        feat_cols = metadata['feature_columns']
        n_samples = 10
        X_dummy = pd.DataFrame(
            np.random.randn(n_samples, len(feat_cols)),
            columns=feat_cols
        )
        
        # Make predictions
        weights = metadata['ensemble_weights']
        
        lgb_px = lgb_x.predict(X_dummy)
        xgb_px = xgb_x.predict(xgb.DMatrix(X_dummy))
        px = weights['lightgbm'] * lgb_px + weights['xgboost'] * xgb_px
        
        lgb_py = lgb_y.predict(X_dummy)
        xgb_py = xgb_y.predict(xgb.DMatrix(X_dummy))
        py = weights['lightgbm'] * lgb_py + weights['xgboost'] * xgb_py
        
        # Verify predictions
        assert len(px) == n_samples
        assert len(py) == n_samples
        assert not np.isnan(px).any()
        assert not np.isnan(py).any()
        assert np.isfinite(px).all()
        assert np.isfinite(py).all()


class TestDeploymentScript:
    """Test deployment automation scripts."""
    
    def test_prepare_for_kaggle_script_exists(self):
        """Verify prepare_for_kaggle.sh exists."""
        script_path = REPO_ROOT / 'scripts' / 'prepare_for_kaggle.sh'
        assert script_path.exists(), "prepare_for_kaggle.sh not found"
    
    def test_full_deployment_script_exists(self):
        """Verify full_deployment.sh exists."""
        script_path = REPO_ROOT / 'scripts' / 'full_deployment.sh'
        assert script_path.exists(), "full_deployment.sh not found"
    
    def test_features_module_exists(self):
        """Verify features.py exists for Kaggle deployment."""
        features_path = REPO_ROOT / 'features.py'
        assert features_path.exists(), "features.py not found in repo root"


class TestTrainingScript:
    """Test the ensemble training script."""
    
    def test_train_ensemble_script_exists(self):
        """Verify train_ensemble.py exists."""
        script_path = REPO_ROOT / 'scripts' / 'train_ensemble.py'
        assert script_path.exists(), "train_ensemble.py not found"
    
    def test_train_ensemble_imports(self):
        """Verify train_ensemble.py imports work."""
        try:
            import lightgbm as lgb
            import xgboost as xgb
        except ImportError:
            pytest.skip("LightGBM or XGBoost not installed")
        
        # Try importing the script (syntax check)
        script_path = REPO_ROOT / 'scripts' / 'train_ensemble.py'
        with open(script_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, str(script_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in train_ensemble.py: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
