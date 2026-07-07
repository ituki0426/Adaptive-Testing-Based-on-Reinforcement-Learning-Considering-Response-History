#!/usr/bin/env Rscript

# Generate uncorrelated 2PL item banks for the simulation study, following the
# settings of Wang, Liu & Xu (2024) with the guessing parameter removed:
#   a ~ N(1.2, 0.25), b ~ N(0, 1), c = 0 (no pseudo-guessing)
# Ten banks of 500 items each are written to data/2PL.
#
# The c column is kept (set to 0) so downstream tools expecting an (a, b, c)
# layout can read 2PL and 3PL banks with the same schema.
#
# NOTE: N(mean, SD) here treats the second argument as the STANDARD DEVIATION,
# matching generate_item_banks.py. If the paper intends a variance, change the
# *_SD constants to sqrt(variance).

# ---- Locate project root (mirrors the MFI script's approach) ----------------
script_path <- if (!is.null(sys.frames()[[1]]$ofile)) {
  normalizePath(sys.frames()[[1]]$ofile)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("--file=", "", args[grep("--file=", args)][1])
  normalizePath(gsub("~\\+~", " ", file_arg))
}
root_dir <- normalizePath(file.path(dirname(script_path), ".."))

# ---- Configuration ----------------------------------------------------------
BANK_COUNT  <- 10L
ITEM_COUNT  <- 500L
RANDOM_SEED <- 20260429L

A_MEAN <- 1.2
A_SD   <- 0.25
B_MEAN <- 0.0
B_SD   <- 1.0

OUTPUT_DIR <- file.path(root_dir, "data", "2PL")

# ---- Truncated sampler ------------------------------------------------------
sample_positive_normal <- function(mean, sd, size) {
  values <- numeric(0)
  while (length(values) < size) {
    draw <- rnorm(size - length(values), mean, sd)
    values <- c(values, draw[draw > 0])
  }
  values[seq_len(size)]
}

# ---- Bank generation --------------------------------------------------------
generate_uncorrelated_bank <- function() {
  a <- sample_positive_normal(A_MEAN, A_SD, ITEM_COUNT)
  b <- rnorm(ITEM_COUNT, B_MEAN, B_SD)
  c <- rep(0, ITEM_COUNT)
  data.frame(a = a, b = b, c = c)
}

# ---- Run --------------------------------------------------------------------
set.seed(RANDOM_SEED)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

for (bank_id in seq_len(BANK_COUNT)) {
  bank <- generate_uncorrelated_bank()
  path <- file.path(OUTPUT_DIR, sprintf("item_bank_uncor_%d.csv", bank_id))
  write.csv(bank, path, row.names = FALSE)
}

message(sprintf("Wrote %d 2PL banks to: %s", BANK_COUNT, OUTPUT_DIR))
