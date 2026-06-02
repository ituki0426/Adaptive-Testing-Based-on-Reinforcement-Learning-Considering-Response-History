#!/usr/bin/env python3
"""
Split theta_true data into train/test sets using the same logic as the DQN notebook
(seed=42, 8:2 split) and export both splits to CSV.

Usage:
    python src/simulation/split_theta.py

Environment variables (all optional):
    BANK_ID      : int,   default 1
    DATA_SIZE    : int,   default 5000  (use 0 for all rows)
    TRAIN_RATIO  : float, default 0.8
    RANDOM_SEED  : int,   default 42
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find project root (no 'data' directory found).")


def get_env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def split_theta(
    bank_id: int,
    data_size: int | None,
    train_ratio: float,
    random_seed: int,
    root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    theta_path = root / "data" / "theta_true" / f"theta_true_{bank_id}.csv"
    theta_all = np.array(pd.read_csv(theta_path)["x"])

    if data_size is not None:
        theta_all = theta_all[: min(data_size, len(theta_all))]

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(theta_all))
    train_size = int(len(theta_all) * train_ratio)
    train_size = min(max(train_size, 1), len(theta_all) - 1)

    theta_train = theta_all[perm[:train_size]]
    theta_test = theta_all[perm[train_size:]]
    return theta_train, theta_test


def main() -> None:
    bank_id     = int(get_env("BANK_ID", "1"))
    data_size_s = get_env("DATA_SIZE", "5000")
    data_size   = None if data_size_s == "0" else int(data_size_s)
    train_ratio = float(get_env("TRAIN_RATIO", "0.8"))
    random_seed = int(get_env("RANDOM_SEED", "42"))

    root = find_project_root()
    theta_train, theta_test = split_theta(bank_id, data_size, train_ratio, random_seed, root)

    out_dir = root / "data" / "theta_true"
    suffix = f"_{bank_id}_seed{random_seed}"
    if data_size is not None:
        suffix += f"_n{data_size}"

    train_path = out_dir / f"theta_train{suffix}.csv"
    test_path  = out_dir / f"theta_test{suffix}.csv"

    pd.DataFrame({"x": theta_train}).to_csv(train_path, index=False)
    pd.DataFrame({"x": theta_test}).to_csv(test_path,  index=False)

    print(f"bank_id    : {bank_id}")
    print(f"data_size  : {data_size if data_size is not None else 'all'}")
    print(f"train_ratio: {train_ratio}")
    print(f"random_seed: {random_seed}")
    print(f"theta_train: {len(theta_train)} rows -> {train_path}")
    print(f"theta_test : {len(theta_test)} rows -> {test_path}")


if __name__ == "__main__":
    main()
