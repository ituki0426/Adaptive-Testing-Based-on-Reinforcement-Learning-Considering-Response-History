# -*- coding: utf-8 -*-
"""Generate 2D 2PL item banks for EXP_v5.

Each item j has:
    a_j1 ~ U(0.5, 2.0)
    a_j2 ~ U(0.5, 2.0)
    b_j  ~ U(-3, 3)

Output layout (default):
    EXP_v5/data/item_banks/item_bank_uncor_{bank_id}.csv
with columns: a1, a2, b (150 rows per bank).
"""

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BANK_COUNT = 10
ITEM_COUNT = 150
RANDOM_SEED = 20260429

A_LOW = 0.5
A_HIGH = 2.0
B_LOW = -3.0
B_HIGH = 3.0


def write_item_bank_files(
    output_dir: Path,
    bank_count: int,
    item_count: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for bank_id in range(1, bank_count + 1):
        rng = np.random.default_rng(seed + bank_id)
        a1 = rng.uniform(A_LOW, A_HIGH, size=item_count)
        a2 = rng.uniform(A_LOW, A_HIGH, size=item_count)
        b = rng.uniform(B_LOW, B_HIGH, size=item_count)

        path = output_dir / f"item_bank_uncor_{bank_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["a1", "a2", "b"])
            for row in zip(a1.tolist(), a2.tolist(), b.tolist()):
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D 2PL item banks for EXP_v5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "item_banks",
        help="Directory for item_bank_uncor_{bank_id}.csv files.",
    )
    parser.add_argument(
        "--bank-count",
        type=int,
        default=BANK_COUNT,
        help="Number of banks to generate.",
    )
    parser.add_argument(
        "--item-count",
        type=int,
        default=ITEM_COUNT,
        help="Number of items per bank.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Base random seed for reproducible generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_item_bank_files(
        output_dir=args.output_dir,
        bank_count=args.bank_count,
        item_count=args.item_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
