# ============================================================================
# h1_sample_size_projection.R
# ----------------------------------------------------------------------------
# Answers: "I currently have N participants and an inconclusive interim BF10.
#           How many MORE do I need before I can expect a conclusive result?"
#
# This is a resampling-based Bayes Factor Design Analysis (BFDA; see
# Schonbrodt & Wagenmakers, 2018, "Bayes factor design analysis: Planning
# for compelling evidence"). It does NOT guarantee a future N -- a Bayes
# factor trajectory is a random walk and can occasionally wander for a long
# time even under a real effect, or hit a lucky BF by chance under the null.
# What it gives you is a probabilistic answer: "if the true effect looks
# like what we've observed in the first N_current participants, X% of
# hypothetical futures reach a conclusive BF10 by N = <this many>."
#
# METHOD
#   1. Take the paired differences (ON - OFF) actually observed so far
#      (h1_participant_level.csv). These stand in for "what a new
#      participant's difference score looks like" -- i.e. the projection
#      assumes future participants resemble the ones already collected.
#   2. Many times (n_sims), bootstrap-resample (with replacement) enough
#      extra differences to reach N = 39 (the protocol's ceiling), append
#      them to the real, observed data, and walk the running two-sided BF10
#      forward one participant at a time.
#   3. Record the first N at which that simulated trajectory becomes
#      conclusive (BF10 > 3 or BF10 < 1/3), if it ever does by N = 39.
#   4. Summarize across simulations: median/typical N needed, and the
#      fraction of simulated futures that are conclusive by each N.
#
# This uses the SAME prior (JZS, rscale = 0.707) and the SAME two-sided test
# as the early-stopping section of h1_analysis.R, so the numbers here are
# directly comparable to what that script reports.
#
# A closed-form BF calculator (BayesFactor::ttest.tstat) is used instead of
# calling ttestBF() on the raw vectors at every step of every simulation --
# it needs only the t-statistic and N, and gives an identical answer, but is
# roughly 100x faster, which matters because this script computes on the
# order of n_sims * (39 - N_current) Bayes factors.
#
# Run it exactly like h1_analysis.R:
#   Rscript h1_sample_size_projection.R [output_folder]
# ============================================================================

library(BayesFactor)

# ---- settings --------------------------------------------------------------

arguments  <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(arguments) >= 1) arguments[1] else "/home/hamzeh/Documents/MEAOW_project/hypothesis_1"

prior_scale <- 0.707   # must match h1_analysis.R
bf_upper    <- 3
bf_lower    <- 1 / 3
n_max       <- 39      # protocol ceiling
n_sims      <- 1000    # number of simulated futures (each BF call costs a
                        # few ms, so n_sims * (39 - N_current) BF calls is
                        # the runtime driver -- 1000 keeps this under ~1 min
                        # in the worst case (N_current close to 22). Raise it
                        # for a less noisy projection if you can wait longer.
set.seed(1)            # remove/change this for a different random draw

participant_file <- file.path(output_dir, "h1_participant_level.csv")

cat(sprintf("Reading from: %s\n", normalizePath(output_dir, mustWork = FALSE)))

if (!file.exists(participant_file)) {
  stop(sprintf(paste0("Cannot find %s\n",
                      "  Run h1_group.py first, or pass the right folder:\n",
                      "      Rscript h1_sample_size_projection.R <output_folder>\n"),
               participant_file))
}

participants <- read.csv(participant_file, stringsAsFactors = FALSE)

required <- c("delta_ON", "delta_OFF")
missing_columns <- setdiff(required, names(participants))
if (length(missing_columns) > 0) {
  stop(sprintf("%s is missing the column(s): %s",
               participant_file, paste(missing_columns, collapse = ", ")))
}

participants <- participants[!is.na(participants$delta_ON) &
                             !is.na(participants$delta_OFF), ]

observed_diffs <- participants$delta_ON - participants$delta_OFF
n_current      <- length(observed_diffs)

if (n_current < 2) {
  stop("Fewer than 2 valid participants -- nothing to project from.")
}

if (n_current >= n_max) {
  cat(sprintf("You already have N = %d, at or above the protocol ceiling of N = %d.\n",
              n_current, n_max))
  cat("There is nothing further to project.\n")
  quit(save = "no", status = 0)
}

# ---- fast closed-form two-sided BF10 from a vector of differences ----------
# Equivalent to ttestBF(x = a one-sample vector of paired differences,
# rscale = prior_scale) -- i.e. the same thing h1_analysis.R's early-stopping
# section is computing (a one-sample/paired JZS t-test), just computed from
# the t-statistic directly instead of re-fitting on the raw vector each time.

bf10_from_diffs <- function(diffs) {
  n <- length(diffs)
  s <- sd(diffs)
  if (n < 2 || s == 0 || is.na(s)) return(NA_real_)
  t_stat <- mean(diffs) / (s / sqrt(n))
  bf <- ttest.tstat(t = t_stat, n1 = n, rscale = prior_scale, simple = TRUE)
  as.numeric(bf)
}

# ---- sanity check: closed-form must match ttestBF() on the real data ------
# This is the "easy sanity check" this project's pipelines are built around:
# if these two numbers disagree, something is wrong with bf10_from_diffs()
# and the rest of this script should not be trusted.

bf_check_fast  <- bf10_from_diffs(observed_diffs)
bf_check_exact <- extractBF(ttestBF(x = observed_diffs, rscale = prior_scale))$bf[1]

cat("\n--- Sanity check: closed-form BF vs. ttestBF() on your real, current data ---\n")
cat(sprintf("Closed-form BF10 (fast, used below) : %.4f\n", bf_check_fast))
cat(sprintf("ttestBF() BF10 (reference)          : %.4f\n", bf_check_exact))
if (abs(bf_check_fast - bf_check_exact) / bf_check_exact > 0.01) {
  stop("Sanity check failed: closed-form BF does not match ttestBF(). Stopping rather than reporting an unverified projection.")
}
cat("Sanity check passed (agree to within 1%).\n\n")

cat(sprintf("Current N = %d.  Observed diff: mean = %.4f, SD = %.4f, current BF10 (two-sided) = %.3f\n\n",
            n_current, mean(observed_diffs), sd(observed_diffs), bf_check_exact))

# ---- simulate forward -------------------------------------------------------

n_to_add <- n_max - n_current

# first_conclusive_n[i] = the N at which simulated future i first became
# conclusive, or NA if it never did by n_max. conclusive_side records which
# way ("alternative" or "null") for the ones that did.
first_conclusive_n <- rep(NA_integer_, n_sims)
conclusive_side    <- rep(NA_character_, n_sims)

for (sim in seq_len(n_sims)) {
  future_diffs <- sample(observed_diffs, size = n_to_add, replace = TRUE)
  running <- observed_diffs

  for (k in seq_len(n_to_add)) {
    running <- c(running, future_diffs[k])
    bf <- bf10_from_diffs(running)
    this_n <- n_current + k

    if (!is.na(bf) && bf > bf_upper) {
      first_conclusive_n[sim] <- this_n
      conclusive_side[sim]    <- "alternative"
      break
    } else if (!is.na(bf) && bf < bf_lower) {
      first_conclusive_n[sim] <- this_n
      conclusive_side[sim]    <- "null"
      break
    }
  }
}

# ---- summarize ---------------------------------------------------------------

cat("============================================================\n")
cat(" PROJECTION: how many MORE participants until a conclusive BF10?\n")
cat(sprintf(" (%d simulated futures, bootstrapped from your %d observed differences,\n",
            n_sims, n_current))
cat(sprintf("  walked forward to the N = %d ceiling)\n", n_max))
cat("============================================================\n\n")

reached  <- !is.na(first_conclusive_n)
pct_any  <- 100 * mean(reached)
cat(sprintf("Simulated futures that reach ANY conclusive BF10 by N = %d: %.1f%%\n",
            n_max, pct_any))

if (any(reached)) {
  cat(sprintf("  -> for the alternative (BF10 > %d): %.1f%% of all sims\n",
              bf_upper, 100 * mean(reached & conclusive_side == "alternative")))
  cat(sprintf("  -> for the null (BF10 < 1/3)       : %.1f%% of all sims\n",
              100 * mean(reached & conclusive_side == "null")))

  n_needed <- first_conclusive_n[reached]
  cat(sprintf("\nAmong sims that DO reach a conclusive result:\n"))
  cat(sprintf("  median N needed     : %d  (%d more participants)\n",
              as.integer(median(n_needed)), as.integer(median(n_needed)) - n_current))
  cat(sprintf("  25th pct N needed   : %d  (%d more)\n",
              as.integer(quantile(n_needed, 0.25)), as.integer(quantile(n_needed, 0.25)) - n_current))
  cat(sprintf("  75th pct N needed   : %d  (%d more)\n",
              as.integer(quantile(n_needed, 0.75)), as.integer(quantile(n_needed, 0.75)) - n_current))
}

cat(sprintf("\nSims still INCONCLUSIVE even at the N = %d ceiling: %.1f%%\n\n",
            n_max, 100 * mean(!reached)))

# Cumulative probability of being conclusive BY each future N -- this is
# usually the more directly useful table: "if I collect up to N = 30, what's
# the chance I'm already done?"
cat(sprintf("%-5s %-14s %-14s %-10s %-10s\n",
            "N", "P(conclusive)", "P(altern.)", "P(null)", "P(still open)"))
for (this_n in (n_current + 1):n_max) {
  by_now        <- reached & (first_conclusive_n <= this_n)
  by_now_alt    <- by_now & conclusive_side == "alternative"
  by_now_null   <- by_now & conclusive_side == "null"
  cat(sprintf("%-5d %-14s %-14s %-10s %-10s\n",
              this_n,
              sprintf("%.1f%%", 100 * mean(by_now)),
              sprintf("%.1f%%", 100 * mean(by_now_alt)),
              sprintf("%.1f%%", 100 * mean(by_now_null)),
              sprintf("%.1f%%", 100 * (1 - mean(by_now)))))
}

results_table <- data.frame(
  N = (n_current + 1):n_max,
  pct_conclusive_by_this_n = sapply((n_current + 1):n_max, function(x) 100 * mean(reached & first_conclusive_n <= x)),
  pct_alternative_by_this_n = sapply((n_current + 1):n_max, function(x) 100 * mean(reached & conclusive_side == "alternative" & first_conclusive_n <= x)),
  pct_null_by_this_n = sapply((n_current + 1):n_max, function(x) 100 * mean(reached & conclusive_side == "null" & first_conclusive_n <= x))
)
write.csv(results_table, file.path(output_dir, "h1_sample_size_projection.csv"), row.names = FALSE)

cat("\nSaved: h1_sample_size_projection.csv\n\n")

cat("IMPORTANT CAVEATS:\n")
cat("1. This assumes future participants resemble the ones you've already\n")
cat("   collected. At N = 25 that assumption itself carries uncertainty --\n")
cat("   treat this as a planning estimate, not a promise.\n")
cat("2. This covers H1 only. Per the protocol, data collection stops only\n")
cat("   when ALL FOUR hypotheses are conclusive, so H1's own projection is\n")
cat("   a lower bound on how much longer the study as a whole may run.\n")
cat("3. This is descriptive/planning information, not a new stopping rule --\n")
cat("   the actual stopping decision still uses the real interim BF10 from\n")
cat("   h1_analysis.R at each true interim look, not this projection.\n")

cat("\nDone.\n")
