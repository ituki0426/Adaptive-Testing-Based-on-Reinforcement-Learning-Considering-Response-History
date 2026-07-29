#!/usr/bin/env Rscript

script_path <- if (!is.null(sys.frames()[[1]]$ofile)) {
  normalizePath(sys.frames()[[1]]$ofile)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  normalizePath(gsub("~\\+~", " ", sub("--file=", "", args[grep("--file=", args)][1])))
}
root_dir <- normalizePath(file.path(dirname(script_path), ".."))
local_r_lib <- file.path(root_dir, ".Rlib")
if (dir.exists(local_r_lib)) {
  .libPaths(c(local_r_lib, .libPaths()))
}

suppressPackageStartupMessages(library(catR))

get_env <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || value == "") default else value
}

estimate_theta <- function(item_bank, item_ids, responses, current_theta, theta_range = c(-4, 4)) {
  administered <- item_bank[item_ids, , drop = FALSE]
  as.numeric(thetaEst(administered, responses, method = "EAP",
                      priorDist = "norm", priorPar = c(0, 1),
                      range = theta_range))
}

run_mfi_cat <- function(item_bank, theta_true, test_length, seed) {
  set.seed(seed)
  testing_size <- length(theta_true)
  records <- vector("list", testing_size * test_length)
  k <- 1

  for (user_id in seq_len(testing_size)) {
    item_ids <- integer(0)
    responses <- integer(0)
    theta_current <- runif(1, -0.5, 0.5)

    for (step in seq_len(test_length)) {
      selected <- nextItem(
        item_bank,
        theta = theta_current,
        out = item_ids,
        criterion = "MFI",
        method = "EAP",
        priorDist = "norm",
        priorPar = c(0, 1),
        range = c(-4, 4)
      )$item

      response <- as.integer(genPattern(theta_true[user_id], item_bank[selected, , drop = FALSE]))
      item_ids <- c(item_ids, selected)
      responses <- c(responses, response)
      theta_current <- estimate_theta(item_bank, item_ids, responses, theta_current)

      records[[k]] <- data.frame(
        userID = user_id,
        step = step,
        itemID = selected,
        resp = response,
        theta_true = theta_true[user_id],
        theta_est = theta_current,
        bias = theta_current - theta_true[user_id]
      )
      k <- k + 1
    }
  }

  do.call(rbind, records)
}

summarize_steps <- function(records) {
  by_step <- split(records, records$step)
  summary <- lapply(by_step, function(x) {
    r <- if (sd(x$theta_true) == 0 || sd(x$theta_est) == 0) {
      NA_real_
    } else {
      cor(x$theta_true, x$theta_est)
    }
    data.frame(
      step = x$step[1],
      Bias = mean(x$bias),
      RMSE = sqrt(mean(x$bias^2)),
      MAE = mean(abs(x$bias)),
      r = r
    )
  })
  do.call(rbind, summary)
}

bank_type <- get_env("BANK_TYPE", "uncor")
bank_id <- as.integer(get_env("BANK_ID", "1"))
test_length <- as.integer(get_env("TEST_LENGTH", "40"))
testing_size <- as.integer(get_env("TESTING_SIZE", "0"))
n_items <- as.integer(get_env("N_ITEMS", "500"))
seed <- as.integer(get_env("SEED", "20260430"))
output_suffix <- get_env("OUTPUT_SUFFIX", "")
theta_csv <- get_env("THETA_CSV", "")
output_dir <- get_env("OUTPUT_DIR", file.path(root_dir, "EXP007", "results"))

bank_dir <- switch(
  bank_type,
  uncor = file.path(root_dir, "data", "uncorrelated_banks"),
  cor = file.path(root_dir, "data", "correlated_banks"),
  stop("BANK_TYPE must be 'uncor' or 'cor'.", call. = FALSE)
)

item_bank_path <- file.path(bank_dir, sprintf("item_bank_%s_%d.csv", bank_type, bank_id))

item_bank <- as.matrix(read.csv(item_bank_path)[, c("a", "b", "c")])
item_bank <- item_bank[seq_len(min(n_items, nrow(item_bank))), , drop = FALSE]
item_bank <- cbind(item_bank, d = 1)

if (nchar(theta_csv) > 0) {
  # Use pre-split theta values (e.g., exported from the DQN notebook).
  theta_true <- read.csv(theta_csv)[["x"]]
  message("Loaded theta from: ", theta_csv, " (n=", length(theta_true), ")")
} else {
  theta_path <- file.path(root_dir, "data", "theta_true", sprintf("theta_true_%d.csv", bank_id))
  theta_true <- read.csv(theta_path)[["x"]]
  if (testing_size > 0) {
    theta_true <- theta_true[seq_len(min(testing_size, length(theta_true)))]
  }
}

if (test_length > nrow(item_bank)) {
  stop("TEST_LENGTH cannot exceed the number of items in the bank.", call. = FALSE)
}

records <- run_mfi_cat(item_bank, theta_true, test_length, seed)
summary_by_step <- summarize_steps(records)

results_dir <- normalizePath(output_dir, mustWork = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

records_path <- file.path(results_dir, sprintf("records_%s_%d_MFI%s.csv", bank_type, bank_id, output_suffix))
summary_path <- file.path(results_dir, sprintf("summary_%s_%d_MFI%s.csv", bank_type, bank_id, output_suffix))

write.csv(records, records_path, row.names = FALSE)
write.csv(summary_by_step, summary_path, row.names = FALSE)

print(tail(summary_by_step, 1))
message("Saved records to: ", records_path)
message("Saved summary to: ", summary_path)
