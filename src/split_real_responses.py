"""Split a real response matrix and its reference theta into train/test sets.

Both files are shuffled with the SAME permutation so that examinee i's responses
stay paired with examinee i's theta. Output filenames match what the EXP027
notebook loads (``real responses for training/testing.csv`` and
``true theta for training/testing.csv``).

Run from the project root:
    python "src/split_real_responses.py"
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "LNIRT_CredentialForm1"
TRAIN_RATIO = 0.7
SEED = 20260430

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / DATASET
RESP_CSV = DATA_DIR / "real responses.csv"
THETA_CSV = DATA_DIR / "true theta.csv"


def main() -> None:
    responses = pd.read_csv(RESP_CSV)
    theta = pd.read_csv(THETA_CSV)

    if len(responses) != len(theta):
        raise ValueError(
            f"Row count mismatch: responses={len(responses)}, theta={len(theta)}."
        )

    n = len(responses)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    train_size = int(n * TRAIN_RATIO)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    outputs = {
        "real responses for training.csv": responses.iloc[train_idx],
        "real responses for testing.csv": responses.iloc[test_idx],
        "true theta for training.csv": theta.iloc[train_idx],
        "true theta for testing.csv": theta.iloc[test_idx],
    }
    for name, frame in outputs.items():
        frame.to_csv(DATA_DIR / name, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"total rows : {n}")
    print(f"train_ratio: {TRAIN_RATIO}  seed: {SEED}")
    print(f"training   : {train_size} rows")
    print(f"testing    : {n - train_size} rows")
    for name in outputs:
        print(f"  wrote {DATA_DIR / name}")


if __name__ == "__main__":
    main()
