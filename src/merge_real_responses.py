"""Merge the LNIRT CredentialForm1 training and testing response CSVs.

Training rows come first, followed by testing rows. Column order and dtypes are
preserved from the source files. A single header row is written.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "LNIRT_CredentialForm1"
TRAINING_CSV = DATA_DIR / "real responses for training.csv"
TESTING_CSV = DATA_DIR / "real responses for testing.csv"
OUTPUT_CSV = DATA_DIR / "real responses.csv"


def main() -> None:
    training = pd.read_csv(TRAINING_CSV)
    testing = pd.read_csv(TESTING_CSV)

    if list(training.columns) != list(testing.columns):
        raise ValueError(
            "Column mismatch between training and testing CSVs.\n"
            f"  training: {list(training.columns)[:5]} ...\n"
            f"  testing : {list(testing.columns)[:5]} ..."
        )

    merged = pd.concat([training, testing], ignore_index=True)
    merged.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"training rows: {len(training)}")
    print(f"testing rows : {len(testing)}")
    print(f"merged rows  : {len(merged)}")
    print(f"columns      : {len(merged.columns)}")
    print(f"saved to     : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
