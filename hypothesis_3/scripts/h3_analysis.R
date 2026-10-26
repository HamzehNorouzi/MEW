#!/usr/bin/env Rscript
# ==============================================================================
# HYPOTHESIS 3 -- post-movement beta ERS, ON vs OFF
# ==============================================================================
#
# Runs the protocol's two statistical parts, in the order the protocol applies
# them:
#
#   PART 1  BAYESIAN EARLY-STOPPING RULE
#           The directional Bayes factor recomputed on the first n participants
#           in enrolment order, for every n from 22 up to the current N.
#           Reports the first n at which it crosses a conclusive boundary
#           (BF10 > 3 = evidence for H3; BF10 < 1/3 = evidence for the
#           null), which direction it crossed, and whether it stayed across.
#           If it never crosses, that is stated explicitly.
#
#   PART 2  CONFIRMATORY TEST OF THE MAIN HYPOTHESIS
#           A paired frequentist t-test (one-tailed, auc_on > auc_off) and a paired
#           Bayesian t-test (JZS Cauchy prior, directional), reported on:
#             (a) the full current sample, and
#             (b) the sample truncated at the stopping N, when PART 1 found a
#                 crossing before the current N.
#
#   SANITY CHECK
#           The last row of the PART 1 trajectory must reproduce the PART 2
#           full-sample Bayes factor exactly, and the row at the stopping N must
#           reproduce the truncated one. If either disagrees the subsetting is
#           wrong and the script stops rather than report an unverified result.
#
#   SAMPLE CAP
#           The protocol plans for at most N = 39. If the input file holds more
#           participants than that, only the first 39 IN ENROLMENT ORDER are
#           analysed and the excluded ids are listed. Both parts above then see
#           the capped sample. Change CONFIG$n_max to alter the limit.
#
# DIRECTION -- H3 predicts a LARGER ERS under stimulation, and ERS is positive, so a larger ERS is a MORE POSITIVE AUC,
#   so the predicted direction is auc_on > auc_off, i.e.
#   diff = auc_on - auc_off > 0.
#
# The interim and the confirmatory test are the SAME directional test, so the
# stopping decision and the final conclusion can never disagree about direction.
#
# Takes no arguments -- edit CONFIG below.
#
#   Rscript h3_analysis.R
#
# Writes into CONFIG$output_dir:
#   h3_sequential_bf.csv   the directional BF at each n from 22 up
#   h3_main_results.csv    the confirmatory test(s)
#
# Requirements: install.packages("BayesFactor")
# ==============================================================================

suppressPackageStartupMessages(library(BayesFactor))

# ---- CONFIG ------------------------------------------------------------------
CONFIG <- list(
  ers_csv           = "/home/hamzeh/Documents/MEAOW_project/hypothesis_3/h3_ers_mean.csv",
  participant_info  = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/participants_info.csv",
  output_dir        = "/home/hamzeh/Documents/MEAOW_project/hypothesis_3",

  id_column         = "participant",
  order_column      = "order",     # enrolment order; see load_order()

  rscale            = 0.707,       # JZS prior scale (Rouder et al., 2009)
  bf_support        = 3,           # BF above this: conclusive for H3
  bf_null           = 1 / 3,       # BF below this: conclusive for the null

  n_min_interim     = 22,          # first n the early-stopping rule checks
  n_max             = 39           # planned maximum sample -- enforced as a
                                   # hard cap on the number of participants
)

# The predicted direction, used by both parts. "greater" means the script tests
# auc_on > auc_off; "less" means it tests auc_on < auc_off.
DIRECTION    <- "greater"
SUPPORT_TEXT <- "supports H3 (larger ERS in ON)"


# ==============================================================================
# The single test used by BOTH parts
# ==============================================================================
#
# Works on the paired differences. A one-sample t-test on the differences is
# arithmetically identical to a paired t-test on the two conditions (same t, df
# and p), and it is what lets PART 1 and PART 2 share one code path -- which is
# what the sanity check at the end verifies.
#
# For the Bayes factor, nullInterval restricts the alternative to the predicted
# direction. extractBF()$bf[1] is that restricted alternative against the point
# null; element 2 is its complement and is deliberately not used.

directional_test <- function(diff, rscale, direction) {
  diff <- diff[!is.na(diff)]
  n <- length(diff)

  # Guard: a Bayes factor is not defined for n < 2 or for zero variance.
  if (n < 2 || !is.finite(stats::sd(diff)) || stats::sd(diff) == 0) {
    return(list(ok = FALSE, n = n, t = NA_real_, df = NA_real_,
                p_one_tailed = NA_real_, ci_bound = NA_real_, cohens_dz = NA_real_,
                mean_diff = if (n >= 1) mean(diff) else NA_real_,
                median_diff = if (n >= 1) stats::median(diff) else NA_real_,
                bf10 = NA_real_))
  }

  freq <- stats::t.test(diff, mu = 0, alternative = direction)
  null_interval <- if (direction == "greater") c(0, Inf) else c(-Inf, 0)
  bf_object <- ttestBF(x = diff, nullInterval = null_interval, rscale = rscale)

  list(
    ok           = TRUE,
    n            = n,
    t            = unname(freq$statistic),
    df           = unname(freq$parameter),
    p_one_tailed = freq$p.value,
    # one-sided CI: lower bound when "greater", upper bound when "less"
    ci_bound     = if (direction == "greater") freq$conf.int[1] else freq$conf.int[2],
    cohens_dz    = mean(diff) / stats::sd(diff),
    mean_diff    = mean(diff),
    median_diff  = stats::median(diff),
    bf10         = extractBF(bf_object)$bf[1]
  )
}

decision_label <- function(bf, config) {
  if (is.na(bf)) return("not computable")
  if (bf > config$bf_support) return(SUPPORT_TEXT)
  if (bf < config$bf_null)    return("supports null")
  "inconclusive"
}

is_conclusive <- function(label) {
  !(label %in% c("inconclusive", "not computable"))
}


# ==============================================================================
# Data
# ==============================================================================

# One row per participant, with ON and OFF as separate columns.
load_wide <- function(path, value_column) {
  if (!file.exists(path)) {
    stop(sprintf("Cannot find %s\n  Run h3_extract.py first, or fix the path in CONFIG.", path))
  }
  long <- utils::read.csv(path, stringsAsFactors = FALSE)

  for (needed in c("participant", "condition", value_column)) {
    if (!(needed %in% names(long))) {
      stop(sprintf("%s has no '%s' column. Columns present: %s",
                   path, needed, paste(names(long), collapse = ", ")))
    }
  }

  # Participant ids are kept as character throughout. If they are left numeric,
  # later name-based lookups silently become POSITION-based lookups and quietly
  # return the wrong rows.
  long$participant <- as.character(long$participant)
  participants <- sort(unique(long$participant))

  on  <- stats::setNames(rep(NA_real_, length(participants)), participants)
  off <- stats::setNames(rep(NA_real_, length(participants)), participants)
  on[long$participant[long$condition == "ON"]]   <- long[[value_column]][long$condition == "ON"]
  off[long$participant[long$condition == "OFF"]] <- long[[value_column]][long$condition == "OFF"]

  wide <- data.frame(participant = participants,
                     auc_on = unname(on), auc_off = unname(off),
                     stringsAsFactors = FALSE)
  complete <- wide[stats::complete.cases(wide), ]

  n_dropped <- nrow(wide) - nrow(complete)
  if (n_dropped > 0) {
    cat(sprintf("%d participant(s) missing one condition, excluded: %s\n",
                n_dropped, paste(setdiff(wide$participant, complete$participant),
                                 collapse = ", ")))
  }
  complete
}

# Enrolment order for the stopping rule. The sequential result depends on this
# order, so when it cannot be established the script says so loudly rather than
# pretending id order is enrolment order.
#
# NOTE read.csv() rewrites column names like "Participant ID" to
# "Participant.ID" unless check.names = FALSE. Matching only the pretty spelling
# silently lost the ordering column and fell back to id order without the
# fallback message being true. Both spellings are accepted here.
load_order <- function(wide, config) {
  fallback <- function(reason) {
    cat(sprintf("ENROLMENT ORDER: %s\n", reason))
    cat("  Falling back to participant-id order. If that is NOT the order people\n")
    cat("  were enrolled in, the stopping-rule trajectory below is not meaningful.\n\n")
    sort(wide$participant)
  }

  if (!file.exists(config$participant_info)) {
    return(fallback(sprintf("%s not found.", config$participant_info)))
  }

  info <- utils::read.csv(config$participant_info, stringsAsFactors = FALSE,
                          check.names = FALSE)

  id_col <- intersect(c(config$id_column, make.names(config$id_column)), names(info))
  if (length(id_col) == 0) {
    return(fallback(sprintf("no '%s' column in %s.",
                            config$id_column, config$participant_info)))
  }
  if (!(config$order_column %in% names(info))) {
    return(fallback(sprintf("no '%s' column in %s.",
                            config$order_column, config$participant_info)))
  }

  ord <- info[[config$order_column]]
  names(ord) <- as.character(info[[id_col[1]]])
  ord <- ord[wide$participant]

  if (any(is.na(ord))) {
    missing_ids <- wide$participant[is.na(ord)]
    return(fallback(sprintf("no enrolment order for: %s.",
                            paste(missing_ids, collapse = ", "))))
  }

  cat(sprintf("ENROLMENT ORDER: taken from '%s' in %s.\n\n",
              config$order_column, basename(config$participant_info)))
  wide$participant[order(ord)]
}

# ==============================================================================
# MAIN
# ==============================================================================

main <- function(config = CONFIG) {
  dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)

  wide <- load_wide(config$ers_csv, "auc")
  n_total <- nrow(wide)

  cat("==============================================================\n")
  cat(" HYPOTHESIS 3 -- post-movement beta ERS, ON vs OFF\n")
  cat("==============================================================\n")
  cat(sprintf("Participants with both conditions: %d\n", n_total))
  cat(sprintf("Predicted direction: auc_on %s auc_off\n",
              if (DIRECTION == "greater") ">" else "<"))
  cat(sprintf("Boundaries: BF10 > %.0f supports H3; BF10 < 1/%.0f supports the null\n\n",
              config$bf_support, 1 / config$bf_null))

  if (n_total < 2) {
    stop("Fewer than 2 participants have both conditions -- no test is possible.")
  }

  enrol_order <- load_order(wide, config)
  wide <- wide[match(enrol_order, wide$participant), ]   # rows now in enrolment order

  # ---- cap at the planned maximum sample ------------------------------------
  # The protocol plans for at most n_max participants. If more have been
  # collected, only the FIRST n_max IN ENROLMENT ORDER are analysed. The cap is
  # applied here, after the ordering above, so that it keeps the people the
  # protocol would have stopped at -- capping before ordering would silently
  # analyse a different subset. Everything below (the stopping rule and both
  # confirmatory tests) therefore sees at most n_max participants.
  if (n_total > config$n_max) {
    excluded <- wide$participant[(config$n_max + 1):n_total]
    cat(sprintf(paste0("CAP: %d participants available, above the planned maximum of %d.\n",
                       "  Analysing the first %d in enrolment order.\n",
                       "  %d excluded: %s\n\n"),
                n_total, config$n_max, config$n_max,
                length(excluded), paste(excluded, collapse = ", ")))
    wide <- wide[seq_len(config$n_max), ]
    n_total <- config$n_max
  }

  diffs_all <- wide$auc_on - wide$auc_off

  # ===========================================================================
  # PART 1 -- BAYESIAN EARLY-STOPPING RULE
  # ===========================================================================

  cat("--------------------------------------------------------------\n")
  cat(" PART 1: Bayesian early-stopping rule\n")
  cat("--------------------------------------------------------------\n")

  sequential <- NULL
  stop_n      <- NA_integer_
  stop_label  <- NA_character_

  if (n_total < config$n_min_interim) {
    cat(sprintf(paste0("N = %d is below the interim minimum of %d.\n",
                       "The stopping rule is NOT yet applicable; data collection continues.\n\n"),
                n_total, config$n_min_interim))
  } else {

    n_values <- config$n_min_interim:n_total
    sequential <- do.call(rbind, lapply(n_values, function(n) {
      r <- directional_test(diffs_all[1:n], config$rscale, DIRECTION)
      data.frame(n = n, bf10 = r$bf10, mean_diff = r$mean_diff,
                 decision = decision_label(r$bf10, config),
                 stringsAsFactors = FALSE)
    }))

    for (i in seq_len(nrow(sequential))) {
      cat(sprintf("  n = %2d   BF10 = %9.3f   mean diff = %9.4f   %s\n",
                  sequential$n[i], sequential$bf10[i],
                  sequential$mean_diff[i], sequential$decision[i]))
    }
    cat("\n")

    utils::write.csv(sequential,
                     file.path(config$output_dir, "h3_sequential_bf.csv"),
                     row.names = FALSE)

    crossings <- sequential[vapply(sequential$decision, is_conclusive, logical(1)), ]

    if (nrow(crossings) > 0) {
      stop_n     <- crossings$n[1]
      stop_label <- crossings$decision[1]
      cat(sprintf("STOPPING RULE: CONCLUSIVE at n = %d  (BF10 = %.3f, %s).\n",
                  stop_n, crossings$bf10[1], stop_label))

      # A Bayes factor trajectory can wander back inside the boundaries. The rule
      # acts on the FIRST crossing, but a later reversal is important to see.
      after <- sequential[sequential$n >= stop_n, ]
      if (all(after$decision == stop_label)) {
        cat(sprintf("  It stayed conclusive in the same direction through n = %d.\n",
                    n_total))
      } else {
        cat(sprintf(paste0("  WARNING: it did NOT stay conclusive -- at n = %d the verdict is '%s'\n",
                           "  (BF10 = %.3f). The rule acts on the first crossing, but this\n",
                           "  instability should be reported alongside it.\n"),
                    n_total, sequential$decision[nrow(sequential)],
                    sequential$bf10[nrow(sequential)]))
      }
    } else {
      final_bf <- sequential$bf10[nrow(sequential)]
      cat(sprintf(paste0("STOPPING RULE: NOT CONCLUSIVE at any n from %d to %d.\n",
                         "  The Bayes factor never crossed %0.0f or 1/%0.0f.\n",
                         "  Final BF10 = %.3f at n = %d -- inconclusive, data collection continues.\n"),
                  config$n_min_interim, n_total,
                  config$bf_support, 1 / config$bf_null, final_bf, n_total))
      cat(sprintf("  It would need to rise by a factor of %.2f to support H3, or fall by a\n",
                  config$bf_support / final_bf))
      cat(sprintf("  factor of %.2f to support the null.\n", final_bf / config$bf_null))
    }

    if (n_total >= config$n_max) {
      cat(sprintf("  NOTE: the planned maximum of N = %d has been reached.\n", config$n_max))
    }
    cat("\n")
  }

  # ===========================================================================
  # PART 2 -- CONFIRMATORY TEST OF THE MAIN HYPOTHESIS
  # ===========================================================================

  cat("--------------------------------------------------------------\n")
  cat(" PART 2: Confirmatory test of the main hypothesis\n")
  cat("--------------------------------------------------------------\n")

  report_sample <- function(tag, idx) {
    v_on  <- wide$auc_on[idx]
    v_off <- wide$auc_off[idx]
    r <- directional_test(v_on - v_off, config$rscale, DIRECTION)

    cat(sprintf("\n[%s]  n = %d\n", tag, r$n))
    if (!r$ok) {
      cat("  Not computable (n < 2 or zero variance).\n")
      return(data.frame(sample = tag, n = r$n, t = NA, df = NA, p_one_tailed = NA,
                        ci_bound_95 = NA, cohens_dz = NA,
                        mean_auc_on = NA, mean_auc_off = NA,
                        median_auc_on = NA, median_auc_off = NA,
                        mean_diff = NA, median_diff = NA, bf10 = NA,
                        decision = "not computable", stringsAsFactors = FALSE))
    }

    cat(sprintf("  auc_on:  mean = %.4f, median = %.4f\n", mean(v_on), stats::median(v_on)))
    cat(sprintf("  auc_off: mean = %.4f, median = %.4f\n", mean(v_off), stats::median(v_off)))
    cat(sprintf("  difference (auc_on - auc_off): mean = %.4f, median = %.4f\n",
                r$mean_diff, r$median_diff))

    cat(sprintf("  Frequentist paired t-test (one-tailed, auc_on > auc_off):\n"))
    cat(sprintf("    t(%d) = %.3f,  p = %.4f,  Cohen's dz = %.3f\n",
                as.integer(r$df), r$t, r$p_one_tailed, r$cohens_dz))
    cat(sprintf("    95%% one-sided CI %s bound = %.4f\n",
                if (DIRECTION == "greater") "lower" else "upper", r$ci_bound))

    label <- decision_label(r$bf10, config)
    cat(sprintf("  Bayesian paired t-test (directional, JZS r = %.3f):\n", config$rscale))
    cat(sprintf("    BF10 = %.3f  -->  %s\n", r$bf10, label))

    freq_sig    <- r$p_one_tailed < 0.05
    bayes_sup   <- r$bf10 > config$bf_support
    if (freq_sig == bayes_sup) {
      cat("  The frequentist and Bayesian tests agree.\n")
    } else {
      cat("  The two approaches DISAGREE. Per the protocol (Dienes, 2021) the\n")
      cat("  conclusion is drawn from the Bayesian test, with p reported alongside.\n")
    }

    data.frame(sample = tag, n = r$n, t = r$t, df = r$df,
               p_one_tailed = r$p_one_tailed, ci_bound_95 = r$ci_bound,
               cohens_dz = r$cohens_dz,
               mean_auc_on = mean(v_on), mean_auc_off = mean(v_off),
               median_auc_on = stats::median(v_on), median_auc_off = stats::median(v_off),
               mean_diff = r$mean_diff, median_diff = r$median_diff,
               bf10 = r$bf10, decision = label, stringsAsFactors = FALSE)
  }

  main_results <- report_sample("full_current_sample", seq_len(n_total))

  if (!is.na(stop_n)) {
    if (stop_n < n_total) {
      cat(sprintf("\n  (The stopping rule first became conclusive at n = %d, so the same\n", stop_n))
      cat("   test is repeated on that truncated sample below.)\n")
      main_results <- rbind(main_results, report_sample("at_stopping_N", seq_len(stop_n)))
    } else {
      cat(sprintf("\n  (The stopping N (%d) equals the current N, so the truncated test\n", stop_n))
      cat("   would be identical to the full-sample test above.)\n")
    }
  }

  cat("\n")
  utils::write.csv(main_results,
                   file.path(config$output_dir, "h3_main_results.csv"),
                   row.names = FALSE)

  # ===========================================================================
  # SANITY CHECK
  # ===========================================================================
  # PART 1's last row and PART 2's full-sample test are the same test on the
  # same people, so their Bayes factors must be identical. If they are not, the
  # sequential subsetting is wrong and nothing above can be trusted.

  cat("--------------------------------------------------------------\n")
  cat(" SANITY CHECK\n")
  cat("--------------------------------------------------------------\n")

  if (is.null(sequential)) {
    cat("No interim trajectory yet (N below the interim minimum) -- nothing to cross-check.\n")
  } else {
    bf_seq_final  <- sequential$bf10[sequential$n == n_total]
    bf_full       <- main_results$bf10[main_results$sample == "full_current_sample"]
    if (!isTRUE(all.equal(bf_seq_final, bf_full, tolerance = 1e-8))) {
      stop(sprintf(paste0("SANITY CHECK FAILED: the interim BF at n = %d (%.6f) does not match\n",
                          "the full-sample confirmatory BF (%.6f). The sequential subsetting is\n",
                          "wrong -- refusing to report an unverified stopping decision."),
                   n_total, bf_seq_final, bf_full))
    }
    cat(sprintf("PASS: interim BF at n = %d matches the full-sample confirmatory BF (%.6f).\n",
                n_total, bf_full))

    if (!is.na(stop_n) && stop_n < n_total) {
      bf_seq_stop <- sequential$bf10[sequential$n == stop_n]
      bf_trunc    <- main_results$bf10[main_results$sample == "at_stopping_N"]
      if (!isTRUE(all.equal(bf_seq_stop, bf_trunc, tolerance = 1e-8))) {
        stop(sprintf(paste0("SANITY CHECK FAILED: the interim BF at the stopping n = %d (%.6f)\n",
                            "does not match the truncated confirmatory BF (%.6f)."),
                     stop_n, bf_seq_stop, bf_trunc))
      }
      cat(sprintf("PASS: interim BF at the stopping n = %d matches its confirmatory BF (%.6f).\n",
                  stop_n, bf_trunc))
    }
  }

  cat("\n")
  cat("IMPORTANT: the protocol stops data collection only when ALL FOUR\n")
  cat("hypotheses have crossed a conclusive boundary. The result above covers\n")
  cat("H3 only and is therefore an interim record, not a stop decision.\n")
  cat("\nDone.\n")

  invisible(list(sequential = sequential, main_results = main_results,
                 stop_n = stop_n))
}

if (sys.nframe() == 0) {
  main()
}
