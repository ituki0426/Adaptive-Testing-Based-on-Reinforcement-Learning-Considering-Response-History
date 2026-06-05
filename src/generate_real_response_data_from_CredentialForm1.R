# -*- coding: utf-8 -*-

required_packages <- c("LNIRT", "mirt")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Missing required R packages: ",
    paste(missing_packages, collapse = ", "),
    ". Install them with install.packages() and rerun this script.",
    call. = FALSE
  )
}

set.seed(20260429)

output_dir <- "data/real_responses_LNIRT_CredentialForm1"
train_ratio <- 0.8

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data("CredentialForm1", package = "LNIRT")

first_scored_item <- which(colnames(CredentialForm1) == "iraw.1")
last_scored_item <- which(colnames(CredentialForm1) == "iraw.170")

if (length(first_scored_item) != 1 || length(last_scored_item) != 1) {
  stop(
    "Could not find scored item columns iraw.1 through iraw.170 in LNIRT::CredentialForm1.",
    call. = FALSE
  )
}

responses <- as.data.frame(
  CredentialForm1[, first_scored_item:last_scored_item]
)
responses[] <- lapply(responses, function(x) as.integer(as.character(x)))
colnames(responses) <- paste0("item_", seq_len(ncol(responses)))

model <- mirt::mirt(responses, 1, itemtype = "3PL", verbose = FALSE)

item_parameters <- mirt::coef(model, IRTpars = TRUE, simplify = TRUE)$items
theta <- as.data.frame(mirt::fscores(model, method = "EAP"))

item_bank <- data.frame(
  a = item_parameters[, "a"],
  b = item_parameters[, "b"],
  c = item_parameters[, "g"],
  row.names = NULL
)

theta <- data.frame(theta = theta[, 1])

all_ids <- sample(seq_len(nrow(responses)))
train_size <- floor(nrow(responses) * train_ratio)
train_ids <- all_ids[seq_len(train_size)]
test_ids <- all_ids[(train_size + 1):nrow(responses)]
test_size <- length(test_ids)

write.csv(
  item_bank,
  file.path(output_dir, "real item bank.csv"),
  row.names = FALSE
)
write.csv(
  responses[train_ids, , drop = FALSE],
  file.path(output_dir, "real responses for training.csv"),
  row.names = FALSE
)
write.csv(
  theta[train_ids, , drop = FALSE],
  file.path(output_dir, "true theta for training.csv"),
  row.names = FALSE
)
write.csv(
  responses[test_ids, , drop = FALSE],
  file.path(output_dir, "real responses for testing.csv"),
  row.names = FALSE
)
write.csv(
  theta[test_ids, , drop = FALSE],
  file.path(output_dir, "true theta for testing.csv"),
  row.names = FALSE
)

message("Wrote CSV files to: ", normalizePath(output_dir))
message("Item count: ", nrow(item_bank))
message("Training examinees: ", train_size)
message("Testing examinees: ", test_size)
