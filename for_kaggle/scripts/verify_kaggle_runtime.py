"""Verify Kaggle runtime compatibility for the inference notebooks.

This script simulates the Kaggle notebook environment by searching for an attached
`for-kaggle*` dataset under `/kaggle/input` or using local repo files. It attempts to:
- locate dataset path and add to `sys.path`
- import `scripts.load_data` and `features`
- load `models/best_model.pkl` via joblib
- read `test.csv` and `test_input.csv`
- run the feature transform on a small sample and run the model predict path

Usage (local):
    python scripts/verify_kaggle_runtime.py

It prints a short report and exits with 0 on success, non-zero on failure.
"""

import sys
from pathlib import Path
import traceback
import joblib
import pandas as pd
import numpy as np


def find_dataset():
    # Look for attached dataset folders under /kaggle/input matching for-kaggle*
    kaggle_root = Path('/kaggle/input')
    if kaggle_root.exists():
        matches = list(kaggle_root.glob('*for-kaggle*'))
        if matches:
            return matches[0]
    # fallback: project-local for_kaggle folder
    local = Path.cwd() / 'for_kaggle'
    if local.exists():
        return local
    return None


def main():
    ok = True
    print('VERIFIER: starting Kaggle runtime simulation')
    ds = find_dataset()
    if ds:
        print('VERIFIER: found dataset path:', ds)
        sys.path.insert(0, str(ds))
    else:
        print('VERIFIER: no for-kaggle dataset found under /kaggle/input or local for_kaggle/')
        print('VERIFIER: falling back to repo root')
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo))
        ds = repo

    print('VERIFIER: sys.path[0] =', sys.path[0])

    # 1) import helpers
    try:
        from scripts.load_data import load_test, load_test_input
        from features import transform_for_inference, add_time_lag_features
        print('VERIFIER: imported scripts.load_data and features OK')
    except Exception as e:
        print('ERROR: failed to import helpers:')
        traceback.print_exc()
        return 2

    # 2) check model
    model_paths = [ds / 'models' / 'best_model.pkl', Path('models') / 'best_model.pkl']
    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = p
            break
    if model_path is None:
        print('ERROR: best_model.pkl not found in:', model_paths)
        return 3
    print('VERIFIER: loading model from', model_path)
    try:
        meta = joblib.load(model_path)
        print('VERIFIER: model loaded; meta keys:', list(meta.keys()) if isinstance(meta, dict) else type(meta))
    except Exception:
        print('ERROR: failed to load model via joblib:')
        traceback.print_exc()
        return 4

    # 3) check test files
    test_input_candidates = [ds / 'test_input.csv', Path('test_input.csv')]
    test_candidates = [ds / 'test.csv', Path('test.csv')]
    test_input_path = next((p for p in test_input_candidates if p.exists()), None)
    test_path = next((p for p in test_candidates if p.exists()), None)
    if test_input_path is None or test_path is None:
        print('ERROR: test_input.csv or test.csv not found. Searched:', test_input_candidates, test_candidates)
        return 5
    print('VERIFIER: test_input:', test_input_path, 'test:', test_path)

    try:
        ti = pd.read_csv(test_input_path)
        t = pd.read_csv(test_path)
        print('VERIFIER: loaded test files shapes:', ti.shape, t.shape)
    except Exception:
        print('ERROR: failed to read test csv files:')
        traceback.print_exc()
        return 6

    # 4) attempt a small transform + predict run
    try:
        # merge a tiny sample
        sample = pd.merge(t.head(50), ti.head(200), on=['game_id','play_id','nfl_id','frame_id'], how='left')
        sample = add_time_lag_features(sample)
        feat_cols = meta.get('feature_columns')
        if feat_cols is None:
            print('ERROR: feature_columns missing from saved model meta')
            return 7
        Xs = transform_for_inference(sample, feat_cols, meta.get('player_position_values', None))
        print('VERIFIER: transformed sample shape:', Xs.shape)
        # load regressors
        mx = meta['models']['x']
        my = meta['models']['y']
        px = mx.predict(Xs)
        py = my.predict(Xs)
        print('VERIFIER: prediction shapes', px.shape, py.shape)
    except Exception:
        print('ERROR: transform/predict failed:')
        traceback.print_exc()
        return 8

    print('\nVERIFIER: all checks passed — inference notebook should run on Kaggle when dataset is attached and Internet is OFF')
    return 0


if __name__ == '__main__':
    sys.exit(main())
