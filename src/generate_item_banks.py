# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import math
import random


ROOT = Path(__file__).resolve().parents[1]
CORRELATED_DIR = ROOT / "data" / "correlated_banks"
UNCORRELATED_DIR = ROOT / "data" / "uncorrelated_banks"

BANK_COUNT = 10
ITEM_COUNT = 500
RANDOM_SEED = 20260429

A_MEAN = 1.2
A_SD = 0.25
B_MEAN = 0.0
B_SD = 1.0
C_MEAN = 0.25
C_SD = 0.02
AB_CORRELATION = 0.5


def sample_positive_normal(rng, mean, sd, size):
    values = []
    while len(values) < size:
        value = rng.gauss(mean, sd)
        if value > 0:
            values.append(value)
    return values


def sample_guessing_parameters(rng, size):
    values = []
    while len(values) < size:
        value = rng.gauss(C_MEAN, C_SD)
        if 0 < value < 1:
            values.append(value)
    return values


def generate_uncorrelated_bank(rng):
    a_values = sample_positive_normal(rng, A_MEAN, A_SD, ITEM_COUNT)
    b_values = [rng.gauss(B_MEAN, B_SD) for _ in range(ITEM_COUNT)]
    c_values = sample_guessing_parameters(rng, ITEM_COUNT)
    return zip(a_values, b_values, c_values)


def generate_correlated_bank(rng):
    rows = []
    c_values = sample_guessing_parameters(rng, ITEM_COUNT)
    residual_scale = math.sqrt(1 - AB_CORRELATION**2)

    while len(rows) < ITEM_COUNT:
        z_a = rng.gauss(0, 1)
        z_b = rng.gauss(0, 1)
        a = A_MEAN + A_SD * z_a
        b = B_MEAN + B_SD * (AB_CORRELATION * z_a + residual_scale * z_b)
        if a > 0:
            rows.append((a, b, c_values[len(rows)]))

    return rows


def write_bank(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "c"])
        writer.writerows(rows)


def write_banks():
    rng = random.Random(RANDOM_SEED)
    CORRELATED_DIR.mkdir(parents=True, exist_ok=True)
    UNCORRELATED_DIR.mkdir(parents=True, exist_ok=True)

    for bank_id in range(1, BANK_COUNT + 1):
        write_bank(CORRELATED_DIR / f"item_bank_cor_{bank_id}.csv", generate_correlated_bank(rng))
        write_bank(
            UNCORRELATED_DIR / f"item_bank_uncor_{bank_id}.csv",
            generate_uncorrelated_bank(rng),
        )


if __name__ == "__main__":
    write_banks()
