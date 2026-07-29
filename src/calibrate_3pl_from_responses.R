# -*- coding: utf-8 -*-
#
# Calibrate a 3PL item bank and a reference (pseudo-true) ability for every
# examinee from a dichotomous response matrix.
#
#   input : data/<dataset>/real responses.csv   (rows = examinees, cols = items, 0/1)
#   output: data/<dataset>/real item bank (3PL).csv   (a, b, c)
#           data/<dataset>/true theta (3PL).csv         (theta, one row per examinee)
#
# The exported theta is the EAP estimate from the FULL response vector. It is a
# pseudo-true reference used to score the CAT experiments, not a latent ground
# truth (real data has no observable true ability).
#
# Why this script exists: a plain 3PL fit on data/LNIRT_CredentialForm1 let the
# difficulty parameters run off to |b| ~ 66 for a few poorly-fitting items,
# which broke the degenerate-response fallback in the CAT code (theta jumped to
# +-33). Here we (1) put priors on the slope and guessing parameters to keep the
# EM from diverging, and (2) winsorize the exported parameters so every item
# stays in a usable range and remains aligned one-to-one with the response
# columns. Item and theta scales come from the same calibration, so they match.
#
# Run from the project root:
#   Rscript "src/calibrate_3pl_from_responses.R"

## ---- configuration -------------------------------------------------------
dataset    <- "LNIRT_CredentialForm1"
input_file <- "real responses.csv"
item_out   <- "real item bank (3PL).csv"
theta_out  <- "true theta (3PL).csv"

use_priors <- TRUE           # stabilise the 3PL EM with priors on a and c
a_bounds   <- c(0.05, 4.0)   # winsorization range for the slope
b_bounds   <- c(-6.0, 6.0)   # winsorization range for the difficulty
c_bounds   <- c(0.0, 0.5)    # winsorization range for the guessing parameter
seed       <- 20260429

## ---- dependencies --------------------------------------------------------
if (!requireNamespace("mirt", quietly = TRUE)) {
  stop(
    "Missing required R package: mirt. ",
    "Install it with install.packages(\"mirt\") and rerun this script.",
    call. = FALSE
  )
}

set.seed(seed)

data_dir  <- file.path("data", dataset)
input_path <- file.path(data_dir, input_file)
if (!file.exists(input_path)) {
  stop("Response file not found: ", input_path, call. = FALSE)
}

## ---- read and validate the response matrix -------------------------------
responses <- read.csv(input_path, check.names = FALSE)
responses[] <- lapply(responses, function(x) as.integer(as.character(x)))

if (anyNA(responses)) {
  stop("Response matrix contains missing or non-integer values.", call. = FALSE)
}
observed <- sort(unique(unlist(responses, use.names = FALSE)))
if (!all(observed %in% c(0L, 1L))) {
  stop(
    "Response matrix must be dichotomous (0/1); found: ",
    paste(observed, collapse = ", "),
    call. = FALSE
  )
}

n_examinees <- nrow(responses)
n_items <- ncol(responses)
message("Read ", input_path)
message("Examinees: ", n_examinees, "  Items: ", n_items)

## ---- calibrate the 3PL model ---------------------------------------------
fit_plain <- function() {
  mirt::mirt(responses, 1, itemtype = "3PL", verbose = FALSE)
}

fit_with_priors <- function() {
  # a1 = slope (log-normal keeps it positive and moderate);
  # g  = lower asymptote / guessing (expbeta shrinks it toward ~0.2).
  model_syntax <- sprintf(
    "F = 1-%d\nPRIOR = (1-%d, a1, lnorm, 0, 0.5), (1-%d, g, expbeta, 1, 4)",
    n_items, n_items, n_items
  )
  mirt::mirt(responses, mirt::mirt.model(model_syntax),
             itemtype = "3PL", verbose = FALSE)
}

model <- NULL
if (use_priors) {
  model <- tryCatch(
    fit_with_priors(),
    error = function(e) {
      message("Prior-based 3PL fit failed (", conditionMessage(e),
              "); falling back to a plain 3PL fit.")
      NULL
    }
  )
}
if (is.null(model)) {
  model <- fit_plain()
}

## ---- extract item parameters ---------------------------------------------
item_parameters <- mirt::coef(model, IRTpars = TRUE, simplify = TRUE)$items
a_raw <- item_parameters[, "a"]
b_raw <- item_parameters[, "b"]
c_raw <- item_parameters[, "g"]

message(sprintf(
  "Raw 3PL parameters: a in [%.3f, %.3f], b in [%.3f, %.3f], c in [%.3f, %.3f]",
  min(a_raw), max(a_raw), min(b_raw), max(b_raw), min(c_raw), max(c_raw)
))

## ---- winsorize and report ------------------------------------------------
a <- pmin(pmax(a_raw, a_bounds[1]), a_bounds[2])
b <- pmin(pmax(b_raw, b_bounds[1]), b_bounds[2])
c <- pmin(pmax(c_raw, c_bounds[1]), c_bounds[2])

changed <- which(a != a_raw | b != b_raw | c != c_raw)
if (length(changed) > 0) {
  message("Winsorized ", length(changed), " item(s):")
  for (i in changed) {
    message(sprintf(
      "  item %d: a %.3f->%.3f, b %.3f->%.3f, c %.3f->%.3f",
      i, a_raw[i], a[i], b_raw[i], b[i], c_raw[i], c[i]
    ))
  }
} else {
  message("No items required winsorization.")
}

item_bank <- data.frame(a = a, b = b, c = c, row.names = NULL)

## ---- score the reference (pseudo-true) abilities -------------------------
# EAP over the full response vector, in the same row order as the input CSV.
theta <- mirt::fscores(model, method = "EAP")[, 1]
theta_df <- data.frame(theta = theta)

message(sprintf(
  "Reference theta: mean=%.3f, sd=%.3f, range=[%.3f, %.3f]",
  mean(theta), stats::sd(theta), min(theta), max(theta)
))

## ---- write outputs (non-destructive names) -------------------------------
item_path  <- file.path(data_dir, item_out)
theta_path <- file.path(data_dir, theta_out)
write.csv(item_bank, item_path, row.names = FALSE)
write.csv(theta_df, theta_path, row.names = FALSE)

message("Wrote item bank : ", normalizePath(item_path))
message("Wrote true theta: ", normalizePath(theta_path))
