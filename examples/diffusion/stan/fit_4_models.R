library(rstan)
library(bridgesampling)
library(jsonlite)

rstan::rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Results are written to:
#   stan/results_4_models/{dataset}/{m0,m1,m2,m3}/bridgesampling.csv
# Full Stan objects remain in {id}.Rdata, and light posterior draws are exported
# to posterior_draws/{id}.csv for Python diagnostics.

args <- commandArgs(trailingOnly = TRUE)
model_arg <- if (length(args) >= 1) tolower(args[1]) else "all"
dataset <- if (length(args) >= 2) args[2] else "empirical"
n_posterior_draws <- if (length(args) >= 3) as.integer(args[3]) else 2048
posterior_seed <- if (length(args) >= 4) as.integer(args[4]) else 2025
data_path_arg <- if (length(args) >= 5) args[5] else ""
results_root_arg <- if (length(args) >= 6) args[6] else ""
resume <- if (length(args) >= 7) as.logical(args[7]) else FALSE

model_specs <- data.frame(
  model = c("m0", "m1", "m2", "m3"),
  model_index = 0:3,
  stringsAsFactors = FALSE
)

num_alpha <- function(model_index) {
  c(1, 2, 2, 4)[model_index + 1]
}

make_init <- function(model_index) {
  n_alpha <- num_alpha(model_index)
  function() {
    list(
      log_alpha = array(rep(0.0, n_alpha), dim = n_alpha),
      log_nu = array(c(0.0, 0.0), dim = 2),
      logit_tau = -2.0
    )
  }
}

export_posterior_draws <- function(stan_samples, model_index, id, output_dir, n_draws, seed) {
  draws <- rstan::extract(
    stan_samples,
    pars = c("log_alpha", "log_nu", "logit_tau"),
    permuted = TRUE
  )
  n_available <- length(draws$logit_tau)
  n_keep <- min(n_draws, n_available)
  set.seed(seed)
  keep <- sort(sample(seq_len(n_available), size = n_keep, replace = FALSE))

  output <- data.frame(
    draw = seq_len(n_keep),
    source_draw = keep,
    id = id,
    assumed_model = paste0("m", model_index),
    stringsAsFactors = FALSE
  )

  for (j in seq_len(num_alpha(model_index))) {
    output[[paste0("alpha_", j - 1)]] <- draws$log_alpha[keep, j]
  }
  for (j in seq_len(2)) {
    output[[paste0("nu_", j - 1)]] <- draws$log_nu[keep, j]
  }
  output[["tau"]] <- draws$logit_tau[keep]

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(
    output,
    file = file.path(output_dir, paste0(id, ".csv")),
    row.names = FALSE
  )
}

write_model_tables <- function(df, results_dir) {
  write.csv(
    df,
    file = file.path(results_dir, "bridgesampling.csv"),
    row.names = FALSE
  )
  write.csv(
    df[, c(
      "id", "mcmc_seed", "mcmc_elapsed_seconds", "bridge_elapsed_seconds",
      "fit_elapsed_seconds", "num_chains", "num_postwarmup_draws",
      "max_rhat", "min_n_eff", "min_n_eff_ratio", "num_divergent",
      "num_max_treedepth", "min_ebfmi", "converged"
    )],
    file = file.path(results_dir, "convergence_diagnostics.csv"),
    row.names = FALSE
  )
}

if (model_arg != "all") {
  stopifnot(model_arg %in% model_specs$model)
  model_specs <- model_specs[model_specs$model == model_arg, , drop = FALSE]
}

command_args <- commandArgs(trailingOnly = FALSE)
file_arg <- command_args[grep("^--file=", command_args)][1]

script_path <- normalizePath(sub("^--file=", "", file_arg))
stan_dir <- dirname(script_path)
base_dir <- dirname(stan_dir)
data_path <- if (nzchar(data_path_arg)) {
  normalizePath(data_path_arg, mustWork = TRUE)
} else {
  file.path(base_dir, "dataset", "json", dataset)
}
stan_file <- file.path(stan_dir, "rdm_4_models.stan")
results_root <- if (nzchar(results_root_arg)) {
  normalizePath(results_root_arg, mustWork = FALSE)
} else {
  file.path(stan_dir, "results_4_models", dataset)
}

data_files <- list.files(data_path, pattern = "\\.json$", full.names = FALSE)
data_files <- sort(data_files)

if (length(data_files) == 0) {
  stop("No JSON files found in: ", data_path)
}

cat("Dataset:", dataset, "\n")
cat("Stan file:", stan_file, "\n")
cat("Models:", paste(model_specs$model, collapse = ", "), "\n\n")
cat("Posterior draws exported per fit:", n_posterior_draws, "\n\n")

stan_model <- rstan::stan_model(stan_file)

for (row in seq_len(nrow(model_specs))) {
  model <- model_specs$model[row]
  model_index <- model_specs$model_index[row]
  results_dir <- file.path(results_root, model)
  posterior_dir <- file.path(results_dir, "posterior_draws")
  parameter_diagnostic_dir <- file.path(results_dir, "parameter_diagnostics")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("Assumed model:", model, "(model_index =", model_index, ")\n")
  bridge_path <- file.path(results_dir, "bridgesampling.csv")
  df <- if (resume && file.exists(bridge_path)) {
    read.csv(bridge_path, stringsAsFactors = FALSE)
  } else {
    data.frame()
  }

  for (i in seq_along(data_files)) {
    data_file <- data_files[i]
    id <- tools::file_path_sans_ext(data_file)
    cached_files <- c(
      file.path(results_dir, paste0(id, ".Rdata")),
      file.path(posterior_dir, paste0(id, ".csv")),
      file.path(parameter_diagnostic_dir, paste0(id, ".csv"))
    )
    cached_columns <- c(
      "id", "estimate", "sd", "mcmc_seed", "mcmc_elapsed_seconds",
      "bridge_elapsed_seconds", "fit_elapsed_seconds", "max_rhat", "min_n_eff",
      "num_divergent", "num_max_treedepth", "min_ebfmi", "converged"
    )
    if (
      resume && all(cached_columns %in% names(df)) &&
      id %in% df$id && all(file.exists(cached_files))
    ) {
      cat(sprintf("Skipping cached %s, %i/%i\n", data_file, i, length(data_files)))
      next
    }
    cat(sprintf("Fitting %s, %i/%i\n", data_file, i, length(data_files)))

    stan_data <- jsonlite::read_json(
      file.path(data_path, data_file),
      simplifyVector = TRUE
    )
    stan_data[["model_index"]] <- model_index

    mcmc_started <- proc.time()[["elapsed"]]
    stan_samples <- rstan::sampling(
      stan_model,
      data = stan_data,
      iter = 12000,
      warmup = 6000,
      chains = 4,
      seed = posterior_seed + 100000 * model_index + i,
      init = make_init(model_index),
      control = list(adapt_delta = 0.995, max_treedepth = 15),
      pars = c("log_alpha", "log_nu", "logit_tau", "alpha", "nu", "tau", "t0")
    )
    mcmc_elapsed_seconds <- proc.time()[["elapsed"]] - mcmc_started

    bridge_started <- proc.time()[["elapsed"]]
    bridge_samples <- bridgesampling::bridge_sampler(
      stan_samples,
      repetitions = 10,
      method = "warp3",
      silent = TRUE
    )
    bridge_elapsed_seconds <- proc.time()[["elapsed"]] - bridge_started

    log_ml <- bridgesampling::logml(bridge_samples)

    fit_summary <- summary(
      stan_samples,
      pars = c("log_alpha", "log_nu", "logit_tau")
    )$summary
    parameter_diagnostics <- data.frame(
      parameter = rownames(fit_summary),
      fit_summary,
      row.names = NULL,
      check.names = FALSE
    )
    dir.create(parameter_diagnostic_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(
      parameter_diagnostics,
      file = file.path(parameter_diagnostic_dir, paste0(id, ".csv")),
      row.names = FALSE
    )

    sampler_params <- rstan::get_sampler_params(stan_samples, inc_warmup = FALSE)
    divergent <- sum(vapply(sampler_params, function(x) sum(x[, "divergent__"]), 0.0))
    treedepth_hits <- sum(vapply(
      sampler_params,
      function(x) sum(x[, "treedepth__"] >= 15),
      0.0
    ))
    ebfmi <- vapply(sampler_params, function(x) {
      energy <- x[, "energy__"]
      mean(diff(energy)^2) / var(energy)
    }, 0.0)
    finite_rhat <- fit_summary[, "Rhat"][is.finite(fit_summary[, "Rhat"])]
    max_rhat <- if (length(finite_rhat)) max(finite_rhat) else NA_real_
    min_n_eff <- min(fit_summary[, "n_eff"], na.rm = TRUE)
    n_postwarmup <- sum(vapply(sampler_params, nrow, 0L))
    converged <- isTRUE(
      divergent == 0 && treedepth_hits == 0 &&
      is.finite(max_rhat) && max_rhat <= 1.01 &&
      is.finite(min(ebfmi)) && min(ebfmi) >= 0.3
    )

    output_row <- match(id, df$id)
    if (is.na(output_row)) output_row <- nrow(df) + 1
    df[output_row, "id"] <- id
    df[output_row, "estimate"] <- log_ml
    df[output_row, "sd"] <- sd(bridge_samples$logml)
    df[output_row, "mcmc_seed"] <- posterior_seed + 100000 * model_index + i
    df[output_row, "mcmc_elapsed_seconds"] <- mcmc_elapsed_seconds
    df[output_row, "bridge_elapsed_seconds"] <- bridge_elapsed_seconds
    df[output_row, "fit_elapsed_seconds"] <- mcmc_elapsed_seconds + bridge_elapsed_seconds
    df[output_row, "num_chains"] <- length(sampler_params)
    df[output_row, "num_postwarmup_draws"] <- n_postwarmup
    df[output_row, "max_rhat"] <- max_rhat
    df[output_row, "min_n_eff"] <- min_n_eff
    df[output_row, "min_n_eff_ratio"] <- min_n_eff / n_postwarmup
    df[output_row, "num_divergent"] <- divergent
    df[output_row, "num_max_treedepth"] <- treedepth_hits
    df[output_row, "min_ebfmi"] <- min(ebfmi)
    df[output_row, "converged"] <- converged

    save(
      stan_samples, bridge_samples, log_ml,
      file = file.path(results_dir, paste0(id, ".Rdata"))
    )
    export_posterior_draws(
      stan_samples,
      model_index = model_index,
      id = id,
      output_dir = posterior_dir,
      n_draws = n_posterior_draws,
      seed = posterior_seed + i
    )
    write_model_tables(df, results_dir)

    cat("Done!\n")
  }

  df <- df[match(tools::file_path_sans_ext(data_files), df$id), , drop = FALSE]
  write_model_tables(df, results_dir)
}
