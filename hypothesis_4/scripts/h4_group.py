# -*- coding: utf-8 -*-
"""
Hypothesis 4, pass 2 -- the fit screen and the participant-level values.

The protocol z-scores the specparam fit metric and drops epochs with z > 2.
That z-score is taken across ALL participants, so it can only be computed once
every epoch exists -- which is why this is a separate pass from h4_extract.py.

One consequence worth knowing: adding participant N+1 changes the group mean and
SD, and therefore can change which epochs survive for everyone else. This script
always rebuilds from h4_epoch_level.csv rather than appending, so re-running it
after new data is the correct and only way to update the results.

Takes no arguments: everything comes from h4_config.py.

    python h4_group.py

Writes into OUTPUT_DIR:
    h4_beta.csv                participant, condition, n_epochs, beta_auc
    h4_fit_report.csv          per participant: epochs dropped by the screen
    h4_group.log
"""

import os
import logging

import numpy as np
import pandas as pd

import h4_config as cfg
from h4_spectra import average, output_tag
from h4_plots import plot_group_spectra

log = logging.getLogger("h4_group")


def setup_logging():
    logfile = os.path.join(cfg.OUTPUT_DIR, "h4_group.log")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    for handler in (logging.FileHandler(logfile, mode="a", encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


def apply_fit_screen(epochs):
    """Add a `keep` column: False where the fit metric is worse than z > 2.

    The z-score is over every epoch of every participant, as section 3.5
    specifies. With "error" as the metric a HIGH value is a poor fit, so the
    screen is one-sided on the upper tail.
    """
    fit = epochs["fit"].to_numpy(dtype=float)
    mean, sd = fit.mean(), fit.std(ddof=0)

    if sd == 0:
        epochs["fit_z"] = 0.0
        epochs["keep"] = True
        return epochs

    epochs["fit_z"] = (fit - mean) / sd
    if cfg.SPECPARAM_FIT_METRIC == "error":
        epochs["keep"] = epochs["fit_z"] <= cfg.FIT_Z_THRESHOLD
    else:
        # r_squared: a LOW value is the poor fit, so the tail flips.
        epochs["keep"] = epochs["fit_z"] >= -cfg.FIT_Z_THRESHOLD
    return epochs


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logging()

    tag = output_tag()
    epoch_csv = os.path.join(cfg.OUTPUT_DIR, "h4_epoch_level%s.csv" % tag)
    if not os.path.exists(epoch_csv):
        log.info("%s not found -- run h4_extract.py first.", epoch_csv)
        return

    epochs = pd.read_csv(epoch_csv)
    epochs = apply_fit_screen(epochs)

    kept = epochs[epochs["keep"]]
    log.info("fit screen: %d of %d epochs dropped at z > %.1f",
             len(epochs) - len(kept), len(epochs), cfg.FIT_Z_THRESHOLD)
    log.info("model fit across retained epochs: %.4f +/- %.4f (mean +/- SD, %s)",
             kept["fit"].mean(), kept["fit"].std(ddof=1), cfg.SPECPARAM_FIT_METRIC)

    # Per-participant record of what the screen removed.
    report = (epochs.groupby("participant")
                    .agg(n_epochs=("keep", "size"),
                         n_dropped=("keep", lambda k: int((~k).sum())),
                         mean_fit=("fit", "mean"))
                    .reset_index())
    report.to_csv(os.path.join(cfg.OUTPUT_DIR, "h4_fit_report%s.csv" % tag),
                  index=False)

    # Participant-level beta power: collapse the surviving epochs per condition.
    rows = []
    for (participant, condition), group in kept.groupby(["participant", "condition"]):
        rows.append({"participant": participant,
                     "condition": condition,
                     "n_epochs": len(group),
                     "beta_auc": float(average(group["beta_auc"].to_numpy()))})

    beta = pd.DataFrame(rows).sort_values(["participant", "condition"])

    # The protocol's 30-trial rule is re-checked here, because the fit screen
    # can push a participant below it after AutoReject had left them above.
    counts = beta.pivot(index="participant", columns="condition", values="n_epochs")
    short = counts[(counts.min(axis=1) < cfg.MIN_TRIALS)].index.tolist()
    incomplete = counts[counts.isna().any(axis=1)].index.tolist()
    dropped = sorted(set(short) | set(incomplete))

    for participant in dropped:
        available = counts.loc[participant]
        log.info("%-8s excluded after the fit screen: ON %s, OFF %s",
                 participant, available.get("ON", "-"), available.get("OFF", "-"))

    beta = beta[~beta["participant"].isin(dropped)]
    if beta.empty:
        log.info("No participants survived the fit screen.")
        return

    beta_csv = os.path.join(cfg.OUTPUT_DIR, "h4_beta%s.csv" % tag)
    beta.to_csv(beta_csv, index=False)
    log.info("%d participants -> %s", beta["participant"].nunique(), beta_csv)

    plot_spectra(kept, dropped, tag)


def plot_spectra(kept, dropped, tag):
    """Group figure of the original and flattened spectra.

    Only the epochs that survived the fit screen are plotted, and only from
    participants who survived the trial-count rule, so the figure shows exactly
    the data the statistics are computed on. Each participant contributes one
    spectrum per condition, collapsed the same way as their beta values, so no
    participant with more epochs counts for more.
    """
    spectra_npz = os.path.join(cfg.OUTPUT_DIR, "h4_epoch_spectra%s.npz" % tag)
    if not os.path.exists(spectra_npz):
        log.info("%s not found -- skipping the spectra figure.", spectra_npz)
        return

    store = np.load(spectra_npz, allow_pickle=False)
    index = {str(key): position for position, key in enumerate(store["epoch_ids"])}

    usable = kept[~kept["participant"].isin(dropped)]
    if "epoch_id" not in usable.columns:
        log.info("h4_epoch_level has no epoch_id column -- re-run h4_extract.py "
                 "to enable the spectra figure.")
        return

    original_by_condition = {"ON": [], "OFF": []}
    flattened_by_condition = {"ON": [], "OFF": []}

    for (participant, condition), group in usable.groupby(["participant", "condition"]):
        positions = [index[key] for key in group["epoch_id"] if key in index]
        if not positions:
            continue
        original_by_condition[condition].append(
            average(store["original"][positions], axis=0))
        flattened_by_condition[condition].append(
            average(store["flattened"][positions], axis=0))

    if not (original_by_condition["ON"] and original_by_condition["OFF"]):
        log.info("Not enough spectra in one condition to plot.")
        return

    out = os.path.join(cfg.OUTPUT_DIR, "h4_group_spectra%s.png" % tag)
    plot_group_spectra(store["freqs"], original_by_condition,
                       flattened_by_condition, average, out)
    log.info("Group spectra -> %s", out)


if __name__ == "__main__":
    main()
