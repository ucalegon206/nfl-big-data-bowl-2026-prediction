"""
Train NFL 2026 player movement prediction models.

Usage:
    python scripts/train_model.py [--max-rows MAX_ROWS] [--output OUTPUT_PATH]

This script trains HistGradientBoostingRegressor models for x and y coordinate
prediction and saves the trained models along with feature metadata.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
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


def train_models(
    X_train: pd.DataFrame,
    yx_train: pd.Series,
    yy_train: pd.Series,
    params_x: Dict[str, Any] = None,
    params_y: Dict[str, Any] = None,
    random_state: int = 42
) -> Tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    """Train x and y coordinate regressors."""
    if params_x is None:
        params_x = {
            'learning_rate': 0.2,
            'max_iter': 400,
            'max_depth': 5,
            'max_bins': 255,
            'min_samples_leaf': 100
        }
    
    if params_y is None:
        params_y = {
            'learning_rate': 0.1,
            'max_iter': 400,
            'max_depth': 8,
            'max_bins': 127,
            'min_samples_leaf': 100
        }
    
    # Train x regressor
    print('Training x-coordinate regressor...')
    mx = HistGradientBoostingRegressor(**params_x, random_state=random_state)
    mx.fit(X_train, yx_train)
    print('✓ x-regressor trained')
    
    # Train y regressor
    print('Training y-coordinate regressor...')
    my = HistGradientBoostingRegressor(**params_y, random_state=random_state)
    my.fit(X_train, yy_train)
    print('✓ y-regressor trained')
    
    return mx, my, params_x, params_y


def evaluate_models(
    mx: HistGradientBoostingRegressor,
    my: HistGradientBoostingRegressor,
    X_val: pd.DataFrame,
    yx_val: pd.Series,
    yy_val: pd.Series
) -> Dict[str, float]:
    """Evaluate models on validation set."""
    px = mx.predict(X_val)
    py = my.predict(X_val)
    
    rmse_x = np.sqrt(mean_squared_error(yx_val, px))
    rmse_y = np.sqrt(mean_squared_error(yy_val, py))
    combined_rmse = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    
    print(f'\nValidation Results:')
    print(f'  RMSE x: {rmse_x:.4f}')
    print(f'  RMSE y: {rmse_y:.4f}')
    print(f'  Combined RMSE: {combined_rmse:.4f}')
    
    # Sanity checks
    print(f'\nPrediction Validation:')
    print(f'  x predictions - min: {px.min():.2f}, max: {px.max():.2f}, mean: {px.mean():.2f}')
    print(f'  y predictions - min: {py.min():.2f}, max: {py.max():.2f}, mean: {py.mean():.2f}')
    print(f'  No NaNs in x: {not np.isnan(px).any()}')
    print(f'  No NaNs in y: {not np.isnan(py).any()}')
    print(f'  All finite x: {np.isfinite(px).all()}')
    print(f'  All finite y: {np.isfinite(py).all()}')
    
    return {
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'combined_rmse': combined_rmse
    }


def save_model(
    mx: HistGradientBoostingRegressor,
    my: HistGradientBoostingRegressor,
    feat_cols: list,
    params_x: Dict[str, Any],
    params_y: Dict[str, Any],
    merged_clean: pd.DataFrame,
    output_path: Path
):
    """Save trained models and metadata.
    
    Saves to both the specified output path and a timestamped version
    for version tracking and cache-busting on Kaggle.
    """
    from datetime import datetime
    
    meta = {
        'feature_columns': feat_cols,
        'models': {'x': mx, 'y': my},
        'best_params': {'x': params_x, 'y': params_y},
        'player_position_values': merged_clean['player_position'].dropna().unique().tolist()
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with NEW versioned naming pattern: nfl_model_v{timestamp}.pkl
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_path = output_path.parent / f"nfl_model_v{timestamp}.pkl"
    
    joblib.dump(meta, versioned_path)
    print(f'\n✓ Model saved to {versioned_path}')
    print(f'  Model size: {versioned_path.stat().st_size / 1024 / 1024:.2f} MB')
    print(f'  Timestamp: {timestamp}')
    print(f'  ✓ NEW naming pattern (nfl_model_v*) to defeat Kaggle caching!')
    
    # Also save as best_model.pkl for backward compatibility (used by prepare_for_kaggle.sh)
    compat_path = output_path.parent / 'best_model.pkl'
    joblib.dump(meta, compat_path)
    print(f'\n✓ Compatibility copy saved to {compat_path}')
    print(f'  (Used by prepare_for_kaggle.sh for packaging)')
    
    # Clean up old versioned models, keeping only the last 5
    pattern = "nfl_model_v[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].pkl"
    old_versions = sorted(output_path.parent.glob(pattern), reverse=True)
    if len(old_versions) > 5:
        for old_file in old_versions[5:]:
            print(f'  Removing old version: {old_file.name}')
            old_file.unlink()
    print(f'  Keeping {min(len(old_versions), 5)} most recent versioned models')


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train NFL 2026 prediction models')
    parser.add_argument('--max-rows', type=int, default=200_000,
                        help='Maximum rows to use for training (default: 200,000)')
    parser.add_argument('--output', type=str, default='models/nfl_model.pkl',
                        help='Output path base for trained model (default: models/nfl_model.pkl)')
    parser.add_argument('--train-dir', type=str, default='train',
                        help='Directory containing training CSV files (default: train)')
    args = parser.parse_args()
    
    train_dir = REPO_ROOT / args.train_dir
    output_path = REPO_ROOT / args.output
    
    print(f'Training configuration:')
    print(f'  Train dir: {train_dir}')
    print(f'  Output: {output_path}')
    print(f'  Max rows: {args.max_rows:,}')
    print()
    
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
    
    # 5. Train models
    mx, my, params_x, params_y = train_models(X_train, yx_train, yy_train)
    
    # 6. Evaluate
    metrics = evaluate_models(mx, my, X_val, yx_val, yy_val)
    
    # 7. Save
    save_model(mx, my, feat_cols, params_x, params_y, merged_clean, output_path)
    
    print('\n✅ Training complete! Model ready for Kaggle submission.')
    return metrics


if __name__ == '__main__':
    main()
