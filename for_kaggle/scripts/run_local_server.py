"""Start a local inference server that loads the trained baseline model and
runs the local gateway to produce `submission.parquet`.

Usage: `python scripts/run_local_server.py` -- assumes model saved at `models/lgbm_baseline.pkl`.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import kaggle_evaluation.nfl_inference_server as nfl_server_module


def _prepare_features_for_inference(df: pd.DataFrame, feature_columns):
    df2 = df.copy()
    df2['dir_rad'] = np.deg2rad(df2['dir'].fillna(0).astype(float))
    df2['dir_sin'] = np.sin(df2['dir_rad'])
    df2['dir_cos'] = np.cos(df2['dir_rad'])
    for col in ['x', 'y', 's', 'a', 'num_frames_output', 'absolute_yardline_number']:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)
        else:
            df2[col] = 0
    if 'player_position' in df2.columns:
        df2['player_pos_code'] = pd.factorize(df2['player_position'])[0]
    else:
        df2['player_pos_code'] = 0
    return df2[feature_columns]


def main(model_path: str = 'models/lgbm_baseline.pkl'):
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model not found at {model_path}; run train/train_lgbm_baseline.py first')

    meta = joblib.load(model_path)
    feature_columns = meta['feature_columns']
    model_x = meta['models']['x']
    model_y = meta['models']['y']

    # Define the predict endpoint that the Gateway expects
    def predict(test_batch, test_input_batch):
        # Convert inputs to pandas DataFrame if needed
        if hasattr(test_batch, 'to_pandas'):
            tb = test_batch.to_pandas()
        else:
            tb = pd.DataFrame(test_batch)

        if hasattr(test_input_batch, 'to_pandas'):
            ti = test_input_batch.to_pandas()
        else:
            ti = pd.DataFrame(test_input_batch)

        # Align features to the rows the gateway expects: join test_batch -> test_input_batch
        keys = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        merged = pd.merge(tb.reset_index(drop=True), ti, on=keys, how='left', suffixes=(None, '_ti'))

        X = _prepare_features_for_inference(merged, feature_columns)
        px = model_x.predict(X)
        py = model_y.predict(X)
        out = pd.DataFrame({'x': px, 'y': py})
        return out

    # Instantiate NFLInferenceServer with our predict function and run the local gateway
    server = nfl_server_module.NFLInferenceServer(predict)
    # Use default data_paths (None) so local files are used
    server.run_local_gateway()


if __name__ == '__main__':
    main()
