"""Minimal production-like inference server using FastAPI.

Provides:
- `GET /health` — simple liveness check
- `POST /predict` — accept a JSON array of rows (or a path to a CSV) and return predictions
- `POST /reload` — reload model from disk (safe to call after model replacement)

This is intentionally lightweight so it can be run behind gunicorn/uvicorn
for production usage.
"""
from pathlib import Path
import logging
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import features as feat

logger = logging.getLogger('prod_server')
logging.basicConfig(level=logging.INFO)

MODEL_PATH_DEFAULT = Path('models/lgbm_baseline.pkl')


class PredictRequest(BaseModel):
    rows: Optional[List[dict]] = None
    csv_path: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[dict]


app = FastAPI(title='NFL Inference')


class ModelWrapper:
    def __init__(self, path: Path = MODEL_PATH_DEFAULT):
        self.path = path
        self.meta = None
        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f'Model file not found: {self.path}')
        self.meta = joblib.load(self.path)
        # Unpack
        self.feature_columns = self.meta['feature_columns']
        self.model_x = self.meta['models']['x']
        self.model_y = self.meta['models']['y']
        self.player_position_values = self.meta.get('player_position_values', [])
        logger.info('Loaded model from %s', self.path)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = feat.transform_for_inference(df, self.feature_columns, self.player_position_values)
        px = self.model_x.predict(X)
        py = self.model_y.predict(X)
        return pd.DataFrame({'x': px, 'y': py})


# Global model instance
MODEL = None


@app.on_event('startup')
def startup_event():
    global MODEL
    try:
        MODEL = ModelWrapper()
    except Exception as e:
        logger.warning('Model failed to load on startup: %s', e)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': MODEL is not None and MODEL.meta is not None}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or MODEL.meta is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    if req.rows:
        df = pd.DataFrame(req.rows)
    elif req.csv_path:
        p = Path(req.csv_path)
        if not p.exists():
            raise HTTPException(status_code=400, detail='csv_path does not exist')
        df = pd.read_csv(p)
    else:
        raise HTTPException(status_code=400, detail='Provide `rows` or `csv_path`')

    preds = MODEL.predict(df)
    out = preds.to_dict(orient='records')
    return PredictResponse(predictions=out)


@app.post('/reload')
def reload_model(path: Optional[str] = None):
    global MODEL
    p = Path(path) if path else MODEL_PATH_DEFAULT
    try:
        MODEL = ModelWrapper(p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'status': 'reloaded', 'path': str(p)}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('scripts.run_production_server:app', host='0.0.0.0', port=8080, log_level='info')
