#!/usr/bin/env Rscript

# Generate uncorrelated 3PL item banks (100 items each) for EXP_v3, following
# the settings of Wang, Liu & Xu (2024):
#   a ~ N(1.2, 0.25), b ~ N(0, 1), c ~ N(0.25, 0.02)
# Ten banks of 100 items each are written to EXP_v3/data/3PL_100items.
#
# NOTE: N(mean, SD) here treats the second argument as the STANDARD DEVIATION,
# matching src/generate_item_banks_3pl.R. The *_SD constants use sqrt(variance)
# to match the paper's specification.

# ---- Locate project root (mirrors the MFI script's approach) ----------------
script_path <- if (!is.null(sys.frames()[[1]]$ofile)) {
  normalizePath(sys.frames()[[1]]$ofile)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("--file=", "", args[grep("--file=", args)][1])
  normalizePath(gsub("~\\+~", " ", file_arg))
}
exp_root <- normalizePath(file.path(dirname(script_path), ".."))

# ---- Configuration ----------------------------------------------------------
BANK_COUNT  <- 10L
ITEM_COUNT  <- 100L
RANDOM_SEED <- 20260429L

A_MEAN <- 1.2
A_SD   <- sqrt(0.25)
B_MEAN <- 0.0
B_SD   <- sqrt(1.0)
C_MEAN <- 0.25
C_SD   <- sqrt(0.02)

OUTPUT_DIR <- file.path(exp_root, "data", "3PL_100items")

# ---- Truncated samplers -----------------------------------------------------
sample_positive_normal <- function(mean, sd, size) {
  values <- numeric(0)
  while (length(values) < size) {
    draw <- rnorm(size - length(values), mean, sd)
    values <- c(values, draw[draw > 0])
  }
  values[seq_len(size)]
}

sample_guessing <- function(size) {
  values <- numeric(0)
  while (length(values) < size) {
    draw <- rnorm(size - length(values), C_MEAN, C_SD)
    values <- c(values, draw[draw > 0 & draw < 1])
  }
  values[seq_len(size)]
}

# ---- Bank generation --------------------------------------------------------
generate_uncorrelated_bank <- function() {
  a <- sample_positive_normal(A_MEAN, A_SD, ITEM_COUNT)
  b <- rnorm(ITEM_COUNT, B_MEAN, B_SD)
  c <- sample_guessing(ITEM_COUNT)
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

message(sprintf("Wrote %d 3PL banks (%d items each) to: %s",
                BANK_COUNT, ITEM_COUNT, OUTPUT_DIR))
