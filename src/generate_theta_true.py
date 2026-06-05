# -*- coding: utf-8 -*-

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BANK_COUNT = 10
EXAMINEE_COUNT = 5000
RANDOM_SEED = 20260429


def write_theta_true_files(output_dir, bank_count, examinee_count, seed):
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bank_id in range(1, bank_count + 1):
        path = output_dir / f"theta_true_{bank_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x"])
            for _ in range(examinee_count):
                writer.writerow([rng.gauss(0.0, 1.0)])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate true examinee ability values for simulated CAT item banks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "theta_true",
        help="Directory for theta_true_<bank_id>.csv files.",
    )
    parser.add_argument(
        "--bank-count",
        type=int,
        default=BANK_COUNT,
        help="Number of theta_true files to generate.",
    )
    parser.add_argument(
        "--examinee-count",
        type=int,
        default=EXAMINEE_COUNT,
        help="Number of examinees per file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_theta_true_files(
        output_dir=args.output_dir,
        bank_count=args.bank_count,
        examinee_count=args.examinee_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
