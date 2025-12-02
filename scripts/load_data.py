"""Simple data-loading utilities for the NFL Big Data Bowl workspace.

These helpers keep things minimal: they concatenate `train/input_*.csv` and
`train/output_*.csv` when requested, and load `test.csv` and `test_input.csv`.

They use pandas for compatibility with downstream scripts.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd


def _concat_csvs(path_glob: str) -> pd.DataFrame:
    # Resolve the pattern relative to this script's repository root when a
    # relative path is provided. This avoids failures when the notebook or
    # process cwd isn't the project root.
    pattern = Path(path_glob)
    if not pattern.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        pattern = repo_root.joinpath(pattern)
    p = pattern.parent
    files = sorted(p.glob(pattern.name))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_all_inputs(train_dir: str = "train") -> pd.DataFrame:
    """Load and concatenate all `input_*.csv` files under `train_dir`.

    Returns an empty DataFrame if no matching files found.
    """
    return _concat_csvs(f"{train_dir}/input_*.csv")


def load_all_outputs(train_dir: str = "train") -> pd.DataFrame:
    """Load and concatenate all `output_*.csv` files under `train_dir`.

    Returns an empty DataFrame if no matching files found.
    """
    return _concat_csvs(f"{train_dir}/output_*.csv")


def load_test_input(base_dir: str = ".") -> pd.DataFrame:
    p = Path(base_dir) / "test_input.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_test(base_dir: str = ".") -> pd.DataFrame:
    p = Path(base_dir) / "test.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


if __name__ == "__main__":
    # quick local sanity check
    print("Loading sample files from current workspace...")
    ti = load_test_input()
    t = load_test()
    print(f"test_input rows: {len(ti)}; test rows: {len(t)}")
