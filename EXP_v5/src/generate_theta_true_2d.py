# -*- coding: utf-8 -*-
"""Generate 2D true examinee ability values for EXP_v5 (2-dim 2PL).

Each theta_i is sampled from MVN(mu, Sigma) with
    mu = (0, 0)
    Sigma = [[1, rho], [rho, 1]]
for rho in {0.0, 0.3, 0.6}.

Output layout (default):
    EXP_v5/data/theta_true/theta_true_rho00_{bank_id}.csv
    EXP_v5/data/theta_true/theta_true_rho03_{bank_id}.csv
    EXP_v5/data/theta_true/theta_true_rho06_{bank_id}.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BANK_COUNT = 10
EXAMINEE_COUNT = 5000
RANDOM_SEED = 20260429
RHOS = (0.0, 0.3, 0.6)


def rho_tag(rho: float) -> str:
    return f"rho{int(round(rho * 10)):02d}"


def write_theta_true_files(
    output_dir: Path,
    bank_count: int,
    examinee_count: int,
    seed: int,
    rhos: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = np.zeros(2)

    for rho in rhos:
        cov = np.array([[1.0, rho], [rho, 1.0]])
        tag = rho_tag(rho)

        for bank_id in range(1, bank_count + 1):
            rng = np.random.default_rng(seed + bank_id * 100 + int(round(rho * 100)))
            samples = rng.multivariate_normal(mean, cov, size=examinee_count)

            path = output_dir / f"theta_true_{tag}_{bank_id}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["theta1", "theta2"])
                writer.writerows(samples.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2D true examinee ability values for EXP_v5 (2-dim 2PL) "
            "under multiple ability correlations rho."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "theta_true",
        help="Directory for theta_true_rho{XX}_{bank_id}.csv files.",
    )
    parser.add_argument(
        "--bank-count",
        type=int,
        default=BANK_COUNT,
        help="Number of banks per rho.",
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
        help="Base random seed for reproducible generation.",
    )
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=list(RHOS),
        help="Ability correlations to generate (default: 0.0 0.3 0.6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_theta_true_files(
        output_dir=args.output_dir,
        bank_count=args.bank_count,
        examinee_count=args.examinee_count,
        seed=args.seed,
        rhos=tuple(args.rhos),
    )


if __name__ == "__main__":
    main()
