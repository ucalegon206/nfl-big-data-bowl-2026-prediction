"""
Train NFL 2026 player movement prediction models using LightGBM + XGBoost ensemble.

Usage:
    python scripts/train_ensemble.py [--max-rows MAX_ROWS] [--output OUTPUT_PATH]

This script trains an ensemble of LightGBM and XGBoost regressors for x and y coordinate
prediction. Models are saved using native formats (no pickle) to avoid serialization issues.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.load_data import load_all_inputs, load_all_outputs
from features import add_time_lag_features, prepare_features


def load_and_merge_training_data(train_dir: Path) -> pd.DataFrame:
    """Load training inputs and outputs, then merge them."""
    print('Loading training inputs and outputs...')
    X = load_all_inputs(train_dir)
    y = load_all_outputs(train_dir)
    
    print(f'Inputs: {len(X):,} rows')
    print(f'Outputs: {len(y):,} rows')
    
    print('Merging training data...')
    merged = X.merge(y, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], 
                     how='inner', suffixes=(None, '_target'))
    print(f'Merged rows: {len(merged):,}')
    return merged


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Apply feature engineering pipeline."""
    print('Adding time-lag features...')
    df = add_time_lag_features(df)
    
    print('Preparing engineered features...')
    feat_df, feat_cols = prepare_features(df)
    
    print(f'Feature columns ({len(feat_cols)}): {feat_cols}')
    print(f'Feature DataFrame shape: {feat_df.shape}')
    return feat_df, feat_cols


def clean_features(feat_df: pd.DataFrame, merged: pd.DataFrame, feat_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows with NaN values in feature columns."""
    print('Cleaning data (removing NaNs)...')
    mask = feat_df[feat_cols].notnull().all(axis=1)
    feat_df_clean = feat_df[mask].reset_index(drop=True)
    merged_clean = merged.loc[mask].reset_index(drop=True)
    
    print(f'Rows after removing NaNs: {len(feat_df_clean):,}')
    print(f'Rows removed: {len(feat_df) - len(feat_df_clean):,}')
    return feat_df_clean, merged_clean


def prepare_train_val_split(
    feat_df_clean: pd.DataFrame,
    merged_clean: pd.DataFrame,
    feat_cols: list,
    max_rows: int = 200_000,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepare training and validation splits, with optional sampling."""
    X_all = feat_df_clean[feat_cols].copy()
    y_x = merged_clean['x_target'].copy()
    y_y = merged_clean['y_target'].copy()
    
    # Sample if dataset is too large
    if len(X_all) > max_rows:
        print(f'Sampling {max_rows} rows for training (from {len(X_all):,})')
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_all), size=max_rows, replace=False)
        X_all = X_all.iloc[idx].reset_index(drop=True)
        y_x = y_x.iloc[idx].reset_index(drop=True)
        y_y = y_y.iloc[idx].reset_index(drop=True)
    
    # Train/val split
    X_train, X_val, yx_train, yx_val, yy_train, yy_val = train_test_split(
        X_all, y_x, y_y, test_size=test_size, random_state=random_state
    )
    
    print(f'Training set: {len(X_train):,} rows')
    print(f'Validation set: {len(X_val):,} rows')
    return X_train, X_val, yx_train, yx_val, yy_train, yy_val


def train_ensemble_models(
    X_train: pd.DataFrame,
    yx_train: pd.Series,
    yy_train: pd.Series,
    X_val: pd.DataFrame,
    yx_val: pd.Series,
    yy_val: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train LightGBM and XGBoost ensemble for x and y coordinates."""
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': random_state
    }
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': random_state,
        'verbosity': 0
    }
    
    models = {}
    
    # Train x-coordinate models
    print('\n' + '='*60)
    print('Training X-Coordinate Models')
    print('='*60)
    
    print('\n[1/4] Training LightGBM for x...')
    lgb_train_x = lgb.Dataset(X_train, label=yx_train)
    lgb_val_x = lgb.Dataset(X_val, label=yx_val, reference=lgb_train_x)
    models['lgb_x'] = lgb.train(
        lgb_params,
        lgb_train_x,
        num_boost_round=500,
        valid_sets=[lgb_val_x],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    print(f'✓ LightGBM x-model trained ({models["lgb_x"].num_trees()} trees)')
    
    print('\n[2/4] Training XGBoost for x...')
    dtrain_x = xgb.DMatrix(X_train, label=yx_train)
    dval_x = xgb.DMatrix(X_val, label=yx_val)
    models['xgb_x'] = xgb.train(
        xgb_params,
        dtrain_x,
        num_boost_round=500,
        evals=[(dval_x, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print(f'✓ XGBoost x-model trained ({models["xgb_x"].num_boosted_rounds()} rounds)')
    
    # Train y-coordinate models
    print('\n' + '='*60)
    print('Training Y-Coordinate Models')
    print('='*60)
    
    print('\n[3/4] Training LightGBM for y...')
    lgb_train_y = lgb.Dataset(X_train, label=yy_train)
    lgb_val_y = lgb.Dataset(X_val, label=yy_val, reference=lgb_train_y)
    models['lgb_y'] = lgb.train(
        lgb_params,
        lgb_train_y,
        num_boost_round=500,
        valid_sets=[lgb_val_y],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    print(f'✓ LightGBM y-model trained ({models["lgb_y"].num_trees()} trees)')
    
    print('\n[4/4] Training XGBoost for y...')
    dtrain_y = xgb.DMatrix(X_train, label=yy_train)
    dval_y = xgb.DMatrix(X_val, label=yy_val)
    models['xgb_y'] = xgb.train(
        xgb_params,
        dtrain_y,
        num_boost_round=500,
        evals=[(dval_y, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print(f'✓ XGBoost y-model trained ({models["xgb_y"].num_boosted_rounds()} rounds)')
    
    return models


def evaluate_ensemble(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    yx_val: pd.Series,
    yy_val: pd.Series,
    lgb_weight: float = 0.5,
    xgb_weight: float = 0.5
) -> Dict[str, float]:
    """Evaluate ensemble on validation set."""
    
    print('\n' + '='*60)
    print('Ensemble Evaluation')
    print('='*60)
    print(f'Weights: LightGBM={lgb_weight:.2f}, XGBoost={xgb_weight:.2f}')
    
    # X predictions
    lgb_px = models['lgb_x'].predict(X_val)
    xgb_px = models['xgb_x'].predict(xgb.DMatrix(X_val))
    px_ensemble = lgb_weight * lgb_px + xgb_weight * xgb_px
    
    # Y predictions
    lgb_py = models['lgb_y'].predict(X_val)
    xgb_py = models['xgb_y'].predict(xgb.DMatrix(X_val))
    py_ensemble = lgb_weight * lgb_py + xgb_weight * xgb_py
    
    # Individual model RMSEs
    lgb_rmse_x = np.sqrt(mean_squared_error(yx_val, lgb_px))
    xgb_rmse_x = np.sqrt(mean_squared_error(yx_val, xgb_px))
    ens_rmse_x = np.sqrt(mean_squared_error(yx_val, px_ensemble))
    
    lgb_rmse_y = np.sqrt(mean_squared_error(yy_val, lgb_py))
    xgb_rmse_y = np.sqrt(mean_squared_error(yy_val, xgb_py))
    ens_rmse_y = np.sqrt(mean_squared_error(yy_val, py_ensemble))
    
    combined_rmse = np.sqrt((ens_rmse_x**2 + ens_rmse_y**2) / 2)
    
    print(f'\nValidation Results (X-coordinate):')
    print(f'  LightGBM RMSE: {lgb_rmse_x:.4f}')
    print(f'  XGBoost RMSE:  {xgb_rmse_x:.4f}')
    print(f'  Ensemble RMSE: {ens_rmse_x:.4f}')
    
    print(f'\nValidation Results (Y-coordinate):')
    print(f'  LightGBM RMSE: {lgb_rmse_y:.4f}')
    print(f'  XGBoost RMSE:  {xgb_rmse_y:.4f}')
    print(f'  Ensemble RMSE: {ens_rmse_y:.4f}')
    
    print(f'\nCombined Ensemble RMSE: {combined_rmse:.4f}')
    
    # Sanity checks
    print(f'\nPrediction Validation:')
    print(f'  x predictions - min: {px_ensemble.min():.2f}, max: {px_ensemble.max():.2f}, mean: {px_ensemble.mean():.2f}')
    print(f'  y predictions - min: {py_ensemble.min():.2f}, max: {py_ensemble.max():.2f}, mean: {py_ensemble.mean():.2f}')
    print(f'  No NaNs in x: {not np.isnan(px_ensemble).any()}')
    print(f'  No NaNs in y: {not np.isnan(py_ensemble).any()}')
    print(f'  All finite x: {np.isfinite(px_ensemble).all()}')
    print(f'  All finite y: {np.isfinite(py_ensemble).all()}')
    
    return {
        'lgb_rmse_x': lgb_rmse_x,
        'xgb_rmse_x': xgb_rmse_x,
        'ensemble_rmse_x': ens_rmse_x,
        'lgb_rmse_y': lgb_rmse_y,
        'xgb_rmse_y': xgb_rmse_y,
        'ensemble_rmse_y': ens_rmse_y,
        'combined_rmse': combined_rmse
    }


def save_ensemble(
    models: Dict[str, Any],
    feat_cols: list,
    merged_clean: pd.DataFrame,
    metrics: Dict[str, float],
    output_path: Path,
    lgb_weight: float = 0.5,
    xgb_weight: float = 0.5
):
    """Save trained ensemble models using native formats (NO PICKLE).
    
    Saves models as:
    - LightGBM: .txt format (text-based booster)
    - XGBoost: .json format (JSON booster)
    - Metadata: .json format
    
    This completely avoids pickle serialization issues.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_path.parent / f"nfl_ensemble_v{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('\n' + '='*60)
    print('Saving Ensemble Models (Native Formats - No Pickle)')
    print('='*60)
    
    # Save LightGBM models as text
    lgb_x_path = output_dir / 'lgb_model_x.txt'
    models['lgb_x'].save_model(str(lgb_x_path))
    print(f'✓ LightGBM x-model saved: {lgb_x_path.name}')
    
    lgb_y_path = output_dir / 'lgb_model_y.txt'
    models['lgb_y'].save_model(str(lgb_y_path))
    print(f'✓ LightGBM y-model saved: {lgb_y_path.name}')
    
    # Save XGBoost models as JSON
    xgb_x_path = output_dir / 'xgb_model_x.json'
    models['xgb_x'].save_model(str(xgb_x_path))
    print(f'✓ XGBoost x-model saved: {xgb_x_path.name}')
    
    xgb_y_path = output_dir / 'xgb_model_y.json'
    models['xgb_y'].save_model(str(xgb_y_path))
    print(f'✓ XGBoost y-model saved: {xgb_y_path.name}')
    
    # Save metadata as JSON
    metadata = {
        'timestamp': timestamp,
        'feature_columns': feat_cols,
        'player_position_values': merged_clean['player_position'].dropna().unique().tolist(),
        'ensemble_weights': {
            'lightgbm': lgb_weight,
            'xgboost': xgb_weight
        },
        'model_files': {
            'lgb_x': lgb_x_path.name,
            'lgb_y': lgb_y_path.name,
            'xgb_x': xgb_x_path.name,
            'xgb_y': xgb_y_path.name
        },
        'metrics': metrics,
        'model_info': {
            'lgb_x_trees': models['lgb_x'].num_trees(),
            'lgb_y_trees': models['lgb_y'].num_trees(),
            'xgb_x_rounds': models['xgb_x'].num_boosted_rounds(),
            'xgb_y_rounds': models['xgb_y'].num_boosted_rounds()
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'✓ Metadata saved: {metadata_path.name}')
    
    # Calculate total size
    total_size = sum(p.stat().st_size for p in output_dir.glob('*') if p.is_file())
    print(f'\n✓ Ensemble saved to: {output_dir}')
    print(f'  Total size: {total_size / 1024 / 1024:.2f} MB')
    print(f'  Timestamp: {timestamp}')
    
    # Create a symlink to latest for convenience
    latest_link = output_path.parent / 'nfl_ensemble_latest'
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)
    print(f'\n✓ Symlink created: {latest_link.name} -> {output_dir.name}')
    
    # Clean up old ensemble directories
    pattern = "nfl_ensemble_v[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"
    import re
    old_dirs = sorted([d for d in output_path.parent.glob('nfl_ensemble_v*') 
                       if d.is_dir() and re.match(pattern, d.name)], 
                      reverse=True)
    removed_count = 0
    for old_dir in old_dirs[1:]:  # Keep the latest (current) one
        import shutil
        shutil.rmtree(old_dir)
        removed_count += 1
    
    if removed_count > 0:
        print(f'\n✓ Cleaned up {removed_count} old ensemble directory(ies)')
    
    return output_dir


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train NFL 2026 ensemble prediction models')
    parser.add_argument('--max-rows', type=int, default=200_000,
                        help='Maximum rows to use for training (default: 200,000)')
    parser.add_argument('--output', type=str, default='models/nfl_ensemble',
                        help='Output directory base for trained models (default: models/nfl_ensemble)')
    parser.add_argument('--train-dir', type=str, default='train',
                        help='Directory containing training CSV files (default: train)')
    parser.add_argument('--lgb-weight', type=float, default=0.5,
                        help='Weight for LightGBM in ensemble (default: 0.5)')
    parser.add_argument('--xgb-weight', type=float, default=0.5,
                        help='Weight for XGBoost in ensemble (default: 0.5)')
    args = parser.parse_args()
    
    train_dir = REPO_ROOT / args.train_dir
    output_path = REPO_ROOT / args.output
    
    print('\n' + '='*60)
    print('NFL 2026 ENSEMBLE MODEL TRAINING')
    print('='*60)
    print(f'Configuration:')
    print(f'  Train dir: {train_dir}')
    print(f'  Output: {output_path}')
    print(f'  Max rows: {args.max_rows:,}')
    print(f'  LightGBM weight: {args.lgb_weight}')
    print(f'  XGBoost weight: {args.xgb_weight}')
    print('='*60)
    
    # 1. Load and merge data
    merged = load_and_merge_training_data(train_dir)
    
    # 2. Engineer features
    feat_df, feat_cols = engineer_features(merged)
    
    # 3. Clean data
    feat_df_clean, merged_clean = clean_features(feat_df, merged, feat_cols)
    
    # 4. Prepare train/val split
    X_train, X_val, yx_train, yx_val, yy_train, yy_val = prepare_train_val_split(
        feat_df_clean, merged_clean, feat_cols, max_rows=args.max_rows
    )
    
    # 5. Train ensemble models
    models = train_ensemble_models(X_train, yx_train, yy_train, X_val, yx_val, yy_val)
    
    # 6. Evaluate ensemble
    metrics = evaluate_ensemble(models, X_val, yx_val, yy_val, 
                                args.lgb_weight, args.xgb_weight)
    
    # 7. Save ensemble
    output_dir = save_ensemble(models, feat_cols, merged_clean, metrics, output_path,
                               args.lgb_weight, args.xgb_weight)
    
    print('\n' + '='*60)
    print('✅ ENSEMBLE TRAINING COMPLETE!')
    print('='*60)
    print(f'Models saved to: {output_dir}')
    print(f'Combined RMSE: {metrics["combined_rmse"]:.4f}')
    print('\nReady for Kaggle deployment (no pickle issues!)')
    print('='*60 + '\n')
    
    return metrics, output_dir


if __name__ == '__main__':
    main()
