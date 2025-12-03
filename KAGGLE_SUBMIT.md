Kaggle Submission — Quick Steps

- Prepare files in the Kaggle notebook working directory or attach as a dataset named e.g. `for_kaggle`.
  - Required files: `models/best_model.pkl`, `test.csv`, `test_input.csv`, `scripts/`, `features.py`.

- Notebook to run: `notebooks/inference_only_submission.ipynb` (minimal inference) or `notebooks/submission_notebook.ipynb`.

- Important: Before running, open **Notebook Settings** and set **Internet = OFF** (Kaggle requires published notebooks to have been executed offline).

- Run: "Restart & Run All" (ensure all cells finish without errors).

- Save & Commit / Publish the executed notebook version.

- Submission: Upload `submission_best_model_OFFICIAL.csv` in the competition Submission UI and select the published notebook version you just created.

- If you get a Kaggle error during submission:
  1. Confirm the published notebook version was executed with Internet disabled.
  2. Confirm the notebook did not import or call network/system modules (avoid `kaggle_evaluation.core`, `socket`, `subprocess` in the published notebook).
  3. If needed, attach screenshots or paste the exact Kaggle error message and the notebook cell number that failed.

Notes
- Use `for_kaggle` dataset to upload project files to the Kaggle runtime. The notebooks already insert `/kaggle/input/for_kaggle` on `sys.path` so imports resolve.
- The `inference_only_submission.ipynb` is minimal and preferred for running on Kaggle (no training, only predict + save CSV).