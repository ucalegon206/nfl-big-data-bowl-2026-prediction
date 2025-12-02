# NFL Big Data Bowl 2026 — Minimal baseline

This workspace contains a simple baseline to validate the evaluation pipeline.

Files added:

- `scripts/load_data.py` — small helpers to load `train/*`, `test.csv`, and `test_input.csv`.
- `train/train_baseline.py` — produces `submission_baseline.csv` by copying `x,y` from `test_input.csv`.
- `requirements.txt` — minimal dependencies.

Quick start (local macOS zsh):

```bash
# create virtualenv (optional)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run baseline to produce `submission_baseline.csv`
python train/train_baseline.py
```

Next steps I can take for you:

- Implement richer feature engineering and a LightGBM / PyTorch model.
- Add a Jupyter notebook with EDA and visualizations.
- Wire the model into the `kaggle_evaluation` `InferenceServer` for local testing.
Production inference
- Run the FastAPI production scaffold (recommended behind gunicorn/uvicorn):

```bash
# development run
python scripts/run_production_server.py

# production run (example with gunicorn + uvicorn workers)
gunicorn -k uvicorn.workers.UvicornWorker -w 4 "scripts.run_production_server:app" -b 0.0.0.0:8080
```

Notes:
- The FastAPI app loads `models/lgbm_baseline.pkl` saved by the training script.
- Use `POST /predict` with JSON `{"rows": [{...}, ...]}` or `{"csv_path": "path/to/file.csv"}`.
- Use `POST /reload` to reload a new model after swapping the file.

Tell me which you'd like next and any compute constraints (GPU/CPU), and I'll continue.
