#!/usr/bin/env Rscript

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]])))
  }

  source_file <- sys.frames()[[1]]$ofile
  if (!is.null(source_file)) {
    return(normalizePath(source_file))
  }

  stop("Could not determine the path of this script.", call. = FALSE)
}

script_path <- get_script_path()
exp_dir <- normalizePath(file.path(dirname(script_path), ".."))
root_dir <- normalizePath(file.path(exp_dir, ".."))
local_r_lib <- file.path(root_dir, ".Rlib")
if (dir.exists(local_r_lib)) {
  .libPaths(c(local_r_lib, .libPaths()))
}

if (!requireNamespace("catR", quietly = TRUE)) {
  stop("The catR package is required. Install it with install.packages('catR').", call. = FALSE)
}

get_env <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || value == "") default else value
}

parse_integer_env <- function(name, default, minimum = 0L) {
  raw_value <- get_env(name, as.character(default))
  value <- suppressWarnings(as.integer(raw_value))
  if (is.na(value) || value < minimum || as.character(value) != raw_value) {
    stop(sprintf("%s must be an integer greater than or equal to %d.", name, minimum), call. = FALSE)
  }
  value
}

resolve_from_root <- function(path, root_dir) {
  if (grepl("^(/|[A-Za-z]:[/\\\\])", path)) {
    return(path)
  }
  file.path(root_dir, path)
}

estimate_theta_mle <- function(item_bank, item_ids, responses, current_theta) {
  if (all(responses == 1L)) {
    return(current_theta + (max(item_bank[, "b"]) - current_theta) / 2)
  }
  if (all(responses == 0L)) {
    return(current_theta - (current_theta - min(item_bank[, "b"])) / 2)
  }

  administered <- item_bank[item_ids, , drop = FALSE]
  as.numeric(catR::thetaEst(
    administered,
    responses,
    method = "ML",
    range = c(-4, 4),
    current.th = current_theta
  ))
}

summarize_steps <- function(theta_true, theta_history, cum_reward_history) {
  test_length <- nrow(theta_history)
  theta_true_sd <- stats::sd(theta_true)

  rows <- lapply(seq_len(test_length), function(step) {
    theta_est <- theta_history[step, ]
    bias <- theta_est - theta_true
    theta_est_sd <- stats::sd(theta_est)
    correlation <- if (
      is.na(theta_true_sd) || theta_true_sd == 0 ||
        is.na(theta_est_sd) || theta_est_sd == 0
    ) {
      NA_real_
    } else {
      stats::cor(theta_true, theta_est)
    }

    data.frame(
      step = step,
      Bias = mean(bias),
      RMSE = sqrt(mean(bias^2)),
      MAE = mean(abs(bias)),
      r = correlation,
      CumReward = mean(cum_reward_history[step, ])
    )
  })

  do.call(rbind, rows)
}

run_mfi_catR <- function(item_bank, theta_true, test_length, seed) {
  set.seed(seed)
  testing_size <- length(theta_true)

  # Draw all initial estimates first, matching the subject-vectorized Python notebook.
  theta_current <- stats::runif(testing_size, -0.5, 0.5)
  item_ids <- matrix(NA_integer_, nrow = test_length, ncol = testing_size)
  responses <- matrix(NA_integer_, nrow = test_length, ncol = testing_size)
  theta_history <- matrix(NA_real_, nrow = test_length, ncol = testing_size)
  cum_reward <- numeric(testing_size)
  cum_reward_history <- matrix(NA_real_, nrow = test_length, ncol = testing_size)

  for (step in seq_len(test_length)) {
    selected <- integer(testing_size)
    step_responses <- integer(testing_size)

    for (user_id in seq_len(testing_size)) {
      administered_ids <- if (step == 1L) {
        NULL
      } else {
        item_ids[seq_len(step - 1L), user_id]
      }

      selected[user_id] <- catR::nextItem(
        item_bank,
        theta = theta_current[user_id],
        out = administered_ids,
        criterion = "MFI",
        method = "ML",
        range = c(-4, 4),
        maxItems = nrow(item_bank)
      )$item

      step_responses[user_id] <- as.integer(catR::genPattern(
        theta_true[user_id],
        item_bank[selected[user_id], , drop = FALSE]
      ))
      cum_reward[user_id] <- cum_reward[user_id] + catR::Ii(
        theta_true[user_id],
        item_bank[selected[user_id], , drop = FALSE]
      )$Ii
    }

    item_ids[step, ] <- selected
    responses[step, ] <- step_responses

    for (user_id in seq_len(testing_size)) {
      theta_current[user_id] <- estimate_theta_mle(
        item_bank,
        item_ids[seq_len(step), user_id],
        responses[seq_len(step), user_id],
        theta_current[user_id]
      )
    }

    theta_history[step, ] <- theta_current
    cum_reward_history[step, ] <- cum_reward
    bias <- theta_current - theta_true
    message(sprintf(
      "step %d, bias %.3f, rmse %.3f, mae %.3f, cum_reward %.3f",
      step,
      mean(bias),
      sqrt(mean(bias^2)),
      mean(abs(bias)),
      mean(cum_reward)
    ))
  }

  records <- data.frame(
    userID = rep(seq_len(testing_size), each = test_length),
    step = rep(seq_len(test_length), times = testing_size),
    itemID = as.vector(item_ids),
    resp = as.vector(responses),
    theta_true = rep(theta_true, each = test_length),
    theta_est = as.vector(theta_history),
    bias = as.vector(theta_history - rep(theta_true, each = test_length)),
    cum_reward = as.vector(cum_reward_history)
  )

  list(
    records = records,
    summary = summarize_steps(theta_true, theta_history, cum_reward_history)
  )
}

# Defaults follow EXP031/notebook/Test_MFI_MLE_on_the_simulated_bank.ipynb,
# with uncorrelated bank 1 selected as requested for this catR version.
bank_type <- get_env("BANK_TYPE", "uncor")
bank_id <- parse_integer_env("BANK_ID", 1L, minimum = 1L)
test_length <- parse_integer_env("TEST_LENGTH", 40L, minimum = 1L)
testing_size <- parse_integer_env("TESTING_SIZE", 0L, minimum = 0L)
n_items <- parse_integer_env("N_ITEMS", 500L, minimum = 1L)
seed <- parse_integer_env("SEED", 20260430L, minimum = 0L)
theta_csv <- get_env("THETA_CSV", "")
output_suffix <- get_env("OUTPUT_SUFFIX", "_catR")
output_dir <- get_env("OUTPUT_DIR", file.path(exp_dir, "results"))

bank_dirs <- c(
  uncor = file.path(root_dir, "data", "uncorrelated_banks"),
  cor = file.path(root_dir, "data", "correlated_banks")
)
if (!bank_type %in% names(bank_dirs)) {
  stop("BANK_TYPE must be either 'uncor' or 'cor'.", call. = FALSE)
}

item_bank_path <- file.path(
  unname(bank_dirs[[bank_type]]),
  sprintf("item_bank_%s_%d.csv", bank_type, bank_id)
)
if (!file.exists(item_bank_path)) {
  stop(sprintf("Item bank not found: %s", item_bank_path), call. = FALSE)
}

item_bank_data <- read.csv(item_bank_path, check.names = FALSE)
required_columns <- c("a", "b", "c")
if (!all(required_columns %in% names(item_bank_data))) {
  stop("The item bank must contain columns a, b, and c.", call. = FALSE)
}
if (n_items > nrow(item_bank_data)) {
  stop("N_ITEMS cannot exceed the number of rows in the item bank.", call. = FALSE)
}
item_bank <- as.matrix(item_bank_data[seq_len(n_items), required_columns, drop = FALSE])
storage.mode(item_bank) <- "double"
item_bank <- cbind(item_bank, d = 1)

if (theta_csv == "") {
  theta_path <- file.path(
    root_dir,
    "data",
    "theta_true",
    sprintf("theta_true_%d.csv", bank_id)
  )
} else {
  theta_path <- resolve_from_root(theta_csv, root_dir)
}
if (!file.exists(theta_path)) {
  stop(sprintf("Theta CSV not found: %s", theta_path), call. = FALSE)
}

theta_data <- read.csv(theta_path, check.names = FALSE)
if (!"x" %in% names(theta_data)) {
  stop("The theta CSV must contain a column named x.", call. = FALSE)
}
theta_true <- as.numeric(theta_data[["x"]])
if (theta_csv == "" && testing_size > 0L) {
  if (testing_size > length(theta_true)) {
    stop("TESTING_SIZE cannot exceed the number of theta values.", call. = FALSE)
  }
  theta_true <- theta_true[seq_len(testing_size)]
}

if (length(theta_true) < 2L) {
  stop("At least two theta values are required to calculate correlation.", call. = FALSE)
}
if (any(!is.finite(theta_true))) {
  stop("All theta values must be finite.", call. = FALSE)
}
if (test_length > nrow(item_bank)) {
  stop("TEST_LENGTH cannot exceed the number of items in the bank.", call. = FALSE)
}

message(sprintf("Item bank : %s (%d items)", item_bank_path, nrow(item_bank)))
message(sprintf("Theta data: %s (%d subjects)", theta_path, length(theta_true)))
message(sprintf(
  "Config    : bank_type=%s, bank_id=%d, test_length=%d, seed=%d",
  bank_type,
  bank_id,
  test_length,
  seed
))

result <- run_mfi_catR(item_bank, theta_true, test_length, seed)

output_dir <- resolve_from_root(output_dir, root_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
stem <- sprintf("%s_%d_MFI_MLE%s", bank_type, bank_id, output_suffix)
records_path <- file.path(output_dir, sprintf("records_%s.csv", stem))
summary_path <- file.path(output_dir, sprintf("summary_%s.csv", stem))

write.csv(result$records, records_path, row.names = FALSE)
write.csv(result$summary, summary_path, row.names = FALSE)

print(tail(result$summary, 1L))
message("Saved records to: ", records_path)
message("Saved summary to: ", summary_path)
