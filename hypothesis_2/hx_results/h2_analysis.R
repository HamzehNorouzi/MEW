#!/usr/bin/env Rscript
# ==============================================================================
# HYPOTHESIS 2 -- preparatory beta ERD, ON vs OFF
# ==============================================================================
#
# Reads h2_erd.csv (written by h2_extract.py) and runs the two tests the
# protocol specifies for H2:
#
#   1. Confirmatory test, on however many participants are currently in
#      h2_erd.csv: a one-tailed paired t-test and a JZS Bayesian paired
#      t-test, both testing whether AUC is more negative in ON than OFF
#      (larger ERD in ON).
#   2. The early-stopping check: the same directional Bayes factor,
#      recomputed after each participant from N = 22 onward, in enrolment
#      order, to see whether/where it has crossed a conclusive threshold
#      (BF > 3 or BF < 1/3).
#
# Takes no arguments -- edit CONFIG below.
#
#   Rscript h2_analysis.R
#
# Writes into CONFIG$output_dir:
#   h2_main_results.csv       the confirmatory test, current sample
#   h2_sequential_bf.csv      the directional BF at each N from 22 up
#
# Requirements: install.packages("BayesFactor")
# ==============================================================================

suppressPackageStartupMessages(library(BayesFactor))

# ---- CONFIG ------------------------------------------------------------------
# Everything you may need to change lives here.
CONFIG <- list(
  erd_csv           = "/home/hamzeh/Documents/MEAOW_project/hypothesis_2/h2_erd.csv",
  participant_info  = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/participants_info.csv",
  output_dir        = "/home/hamzeh/Documents/MEAOW_project/hypothesis_2",

  id_column         = "Participant ID",
  order_column      = "order",     # enrolment order; optional -- see load_order()

  rscale            = 0.707,       # JZS prior scale (Rouder et al., 2009)
  bf_support        = 3,           # BF above this: conclusive for the directional H
  bf_null           = 1 / 3,       # BF below this: conclusive for the null

  n_min_interim     = 22,          # first N the early-stopping rule checks
  n_max             = 39           # planned maximum sample
)


# ---- Data --------------------------------------------------------------------

# One row per participant, with ON and OFF AUC as separate columns.
load_wide_auc <- function(path) {
  long <- read.csv(path, stringsAsFactors = FALSE)
  participants <- sort(unique(long$participant))

  on  <- setNames(rep(NA_real_, length(participants)), participants)
  off <- setNames(rep(NA_real_, length(participants)), participants)
  on[long$participant[long$condition == "ON"]]   <- long$auc[long$condition == "ON"]
  off[long$participant[long$condition == "OFF"]] <- long$auc[long$condition == "OFF"]

  wide <- data.frame(participant = participants, auc_on = unname(on),
                     auc_off = unname(off), stringsAsFactors = FALSE)
  complete <- wide[complete.cases(wide), ]

  n_dropped <- nrow(wide) - nrow(complete)
  if (n_dropped > 0) {
    cat(sprintf("%d participant(s) missing one condition, excluded: %s\n",
               n_dropped, paste(setdiff(wide$participant, complete$participant),
                                collapse = ", ")))
  }
  complete
}

# Enrolment order for the sequential check. Uses CONFIG$order_column from
# participant_info.csv when present; otherwise falls back to sorting the
# participant id, and says so -- the sequential result depends on this order.
load_order <- function(wide, config) {
  info <- read.csv(config$participant_info, stringsAsFactors = FALSE)

  if (config$order_column %in% names(info)) {
    ord <- info[[config$order_column]]
    names(ord) <- info[[config$id_column]]
    ord <- ord[wide$participant]
    if (!any(is.na(ord))) {
      return(wide$participant[order(ord)])
    }
    cat("Some participants have no enrolment order in", config$participant_info,
       "-- falling back to sorting by participant id.\n")
  } else {
    cat("No '", config$order_column, "' column in ", config$participant_info,
       " -- assuming participants were enrolled in id order.\n", sep = "")
  }
  sort(wide$participant)
}


# ---- One directional test, given a set of ON/OFF pairs -----------------------

# H2 is directional: ERD is negative, and a LARGER ERD is a MORE NEGATIVE AUC.
# So the predicted direction is auc_on < auc_off, i.e. diff = auc_on - auc_off < 0.
directional_test <- function(auc_on, auc_off, rscale) {
  diff <- auc_on - auc_off

  freq <- t.test(auc_on, auc_off, paired = TRUE, alternative = "less")

  bf_object <- ttestBF(x = diff, nullInterval = c(-Inf, 0), rscale = rscale)
  bf10 <- extractBF(bf_object)$bf[1]     # BF for "diff < 0" vs the point null

  list(
    n = length(diff),
    t = unname(freq$statistic),
    df = unname(freq$parameter),
    p_one_tailed = freq$p.value,
    ci_upper = freq$conf.int[2],         # one-sided CI: (-Inf, ci_upper)
    cohens_dz = mean(diff) / sd(diff),
    median_on = median(auc_on),
    median_off = median(auc_off),
    median_diff = median(diff),
    bf10 = bf10
  )
}

decision_label <- function(bf, bf_support, bf_null) {
  if (bf > bf_support) return("supports H2 (larger ERD in ON)")
  if (bf < bf_null) return("supports null")
  "inconclusive"
}


# ---- Main ----------------------------------------------------------------

main <- function(config = CONFIG) {
  dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)

  wide <- load_wide_auc(config$erd_csv)
  n_total <- nrow(wide)

  # ---- 1. Confirmatory test on the current sample -------------------------
  result <- directional_test(wide$auc_on, wide$auc_off, config$rscale)

  main_results <- data.frame(
    n = result$n,
    t = result$t,
    df = result$df,
    p_one_tailed = result$p_one_tailed,
    ci_upper_95 = result$ci_upper,
    cohens_dz = result$cohens_dz,
    median_auc_on = result$median_on,
    median_auc_off = result$median_off,
    median_diff = result$median_diff,
    bf10 = result$bf10,
    decision = decision_label(result$bf10, config$bf_support, config$bf_null)
  )
  write.csv(main_results, file.path(config$output_dir, "h2_main_results.csv"),
           row.names = FALSE)

  cat(sprintf("N = %d   t(%d) = %.2f   p = %.4f (one-tailed)   BF = %.2f   %s\n",
             result$n, result$df, result$t, result$p_one_tailed, result$bf10,
             main_results$decision))
  cat(sprintf("median AUC: ON %.2f, OFF %.2f, diff %.2f\n",
             result$median_on, result$median_off, result$median_diff))

  # ---- 2. Early-stopping check, N = n_min_interim upward -------------------
  if (n_total < config$n_min_interim) {
    cat(sprintf("N = %d is below the interim minimum (%d); no sequential check yet.\n",
               n_total, config$n_min_interim))
    return(invisible(main_results))
  }
  if (n_total > config$n_max) {
    cat(sprintf("N = %d exceeds the planned maximum (%d).\n", n_total, config$n_max))
  }

  enrol_order <- load_order(wide, config)
  rownames(wide) <- wide$participant

  n_values <- config$n_min_interim:n_total
  sequential <- do.call(rbind, lapply(n_values, function(n) {
    ids <- enrol_order[1:n]
    r <- directional_test(wide[ids, "auc_on"], wide[ids, "auc_off"], config$rscale)
    data.frame(n = n, bf10 = r$bf10,
              decision = decision_label(r$bf10, config$bf_support, config$bf_null))
  }))
  write.csv(sequential, file.path(config$output_dir, "h2_sequential_bf.csv"),
           row.names = FALSE)

  conclusive <- sequential[sequential$decision != "inconclusive", ]
  if (nrow(conclusive) > 0) {
    first <- conclusive[1, ]
    cat(sprintf("Conclusive at N = %d: BF = %.2f, %s\n",
               first$n, first$bf10, first$decision))
  } else {
    cat(sprintf("Not yet conclusive at N = %d (BF = %.2f).\n",
               n_total, sequential$bf10[nrow(sequential)]))
  }

  invisible(list(main_results = main_results, sequential = sequential))
}

if (sys.nframe() == 0) {
  main()
}
