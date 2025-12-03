import os
from pathlib import Path
import joblib
import numpy as np
import pytest

from scripts.load_data import load_test_input, load_test
from features import prepare_features, add_time_lag_features, transform_for_inference


def test_model_file_exists():
    p = Path('models/lgbm_baseline.pkl')
    assert p.exists(), "Model file 'models/lgbm_baseline.pkl' not found — run training first"


def test_model_meta_and_basic_predict():
    meta = joblib.load('models/lgbm_baseline.pkl')
    assert isinstance(meta, dict), 'Model metadata must be a dict'
    assert 'feature_columns' in meta and 'models' in meta, 'Missing required keys in model metadata'
    assert 'x' in meta['models'] and 'y' in meta['models'], 'Expected regressors for x and y in meta["models"]'

    # Load a small sample from test_input for a fast, deterministic smoke test
    ti = load_test_input('.')
    assert len(ti) > 0, 'test_input appears empty'
    sample = ti.head(500).copy()

    # add lag features (should be a no-op or add columns safely)
    sample = add_time_lag_features(sample)

    # compute features and confirm required columns exist
    feat_df, feat_cols = prepare_features(sample)
    for c in meta['feature_columns']:
        assert c in feat_df.columns, f"Feature column '{c}' missing after feature preparation"

    # transform for inference and run predictions
    X_eval = transform_for_inference(sample, meta['feature_columns'], meta.get('player_position_values', None))
    mx = meta['models']['x']
    my = meta['models']['y']
    px = mx.predict(X_eval)
    py = my.predict(X_eval)

    # sanity checks
    assert len(px) == len(X_eval) and len(py) == len(X_eval)
    assert not np.isnan(px).any(), 'Predicted x contains NaNs'
    assert not np.isnan(py).any(), 'Predicted y contains NaNs'

    # bounds check (allowing some slack): x ~ [ -10, 130 ], y ~ [ -10, 100 ]
    assert ((px >= -10) & (px <= 130)).all(), 'Predicted x out of expected bounds'
    assert ((py >= -10) & (py <= 100)).all(), 'Predicted y out of expected bounds'
