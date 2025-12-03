"""Feature engineering helpers for training and inference.

Expose small, well-tested functions so training and production use the same
transformations.
"""
from typing import List
import numpy as np
import pandas as pd


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
    return df


def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    """Prepare a DataFrame with feature columns and return (df, feature_cols).

    The function is intentionally conservative: missing columns are filled with 0.
    Use the same feature list used in the training script for consistency.
    """
    df = df.copy()
    df = _ensure_numeric(df, ['x', 'y', 's', 'a', 'num_frames_output', 'absolute_yardline_number', 'frame_id', 'ball_land_x', 'ball_land_y'])
    # direction -> sin/cos
    df['dir'] = df.get('dir', 0)
    df['dir_rad'] = np.deg2rad(pd.to_numeric(df['dir'], errors='coerce').fillna(0))
    df['dir_sin'] = np.sin(df['dir_rad'])
    df['dir_cos'] = np.cos(df['dir_rad'])

    # simple player position encoding - keep deterministic order from unique values
    if 'player_position' in df.columns:
        positions = pd.Categorical(df['player_position'])
        df['player_pos_code'] = positions.codes
    else:
        df['player_pos_code'] = 0

    # Add ball-relative features when available
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        df['dx_ball'] = df['x'] - df['ball_land_x']
        df['dy_ball'] = df['y'] - df['ball_land_y']
        df['dist_ball'] = np.hypot(df['dx_ball'], df['dy_ball'])
    else:
        df['dx_ball'] = 0
        df['dy_ball'] = 0
        df['dist_ball'] = 0

    feature_cols = [
        'x',
        'y',
        's',
        'a',
        'dir_sin',
        'dir_cos',
        'num_frames_output',
        'absolute_yardline_number',
        'player_pos_code',
        'dx_ball',
        'dy_ball',
        'dist_ball',
    ]
    return df, feature_cols


def transform_for_inference(df: pd.DataFrame, feature_columns: List[str], player_position_values: List[str] = None) -> pd.DataFrame:
    """Apply transformations used at training to a new DataFrame and return feature matrix.

    If `player_position_values` is supplied, it is used to align categorical codes to training mapping.
    """
    df = df.copy()
    df, _ = prepare_features(df)
    # If provided, map player_position to codes based on training values
    if player_position_values is not None and 'player_position' in df.columns:
        mapping = {v: i for i, v in enumerate(player_position_values)}
        df['player_pos_code'] = df['player_position'].map(mapping).fillna(-1).astype(int)
    # Ensure feature columns exist
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    return df[feature_columns]


def add_time_lag_features(df: pd.DataFrame, group_keys: List[str] = None, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """Add lag/delta features per group (e.g. per player in a play).

    Produces features like `dx_lag_1` = x - x.shift(1) and `vx_lag_1` = (x - x.shift(1))/frame_delta.
    group_keys defaults to `['game_id','play_id','nfl_id']` if not provided.
    """
    if group_keys is None:
        group_keys = ['game_id', 'play_id', 'nfl_id']
    df = df.copy()
    df = _ensure_numeric(df, ['frame_id', 'x', 'y', 's', 'a'])
    df = df.sort_values(group_keys + ['frame_id'])
    grp = df.groupby(group_keys)
    for lag in lags:
        df[f'x_shift_{lag}'] = grp['x'].shift(lag)
        df[f'y_shift_{lag}'] = grp['y'].shift(lag)
        df[f'frame_shift_{lag}'] = grp['frame_id'].shift(lag)
        df[f'dx_lag_{lag}'] = df['x'] - df[f'x_shift_{lag}']
        df[f'dy_lag_{lag}'] = df['y'] - df[f'y_shift_{lag}']
        # time delta (frames) - guard divide-by-zero
        df[f'dt_lag_{lag}'] = (df['frame_id'] - df[f'frame_shift_{lag}']).replace(0, np.nan)
        df[f'vx_lag_{lag}'] = df[f'dx_lag_{lag}'] / df[f'dt_lag_{lag}']
        df[f'vy_lag_{lag}'] = df[f'dy_lag_{lag}'] / df[f'dt_lag_{lag}']
        # fillna with 0 for the new features
        df[[f'dx_lag_{lag}', f'dy_lag_{lag}', f'dt_lag_{lag}', f'vx_lag_{lag}', f'vy_lag_{lag}']] = df[[f'dx_lag_{lag}', f'dy_lag_{lag}', f'dt_lag_{lag}', f'vx_lag_{lag}', f'vy_lag_{lag}']].fillna(0)
    return df

