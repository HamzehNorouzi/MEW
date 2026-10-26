"""
h1_group.py
------------
Hypothesis 1, panel B of the flowchart: group-level assembly.

This script reads h1_epoch_level.csv and:
    1. z-scores the FOOOF fit ERROR across ALL participants and drops
       every epoch with z > 2
    2. averages the surviving epochs into one value per
       participant x session x condition x ROI
    3. keeps the contralateral ROI for each participant
    4. computes the change scores
           delta_ON  = post_ON  - pre_ON
           delta_OFF = post_OFF - pre_OFF
           diff      = delta_ON - delta_OFF
    5. writes the table that R will analyse
    6. rebuilds the same table for every interim sample size N from 22
       upward, so the sequential Bayes factors can be computed in R

Why step 6 repeats the whole thing: because the z-score is taken across
participants, adding one participant changes the group mean and SD, which
can change which epochs survive for everybody. The interim samples must
therefore be rebuilt from scratch, not appended to.

Output: h1_participant_level.csv
        h1_sequential_input.csv
        h1_specparam_report.csv
        (appends to h1_run.log)

Run it with:  python h1_group.py
"""

import os
import logging

import numpy as np
import pandas as pd

import h1_config as cfg
from h1_extract import setup_logging, find_column, normalise_participant_id


SESSIONS = ["pre_task_rest", "post_task_rest"]
CONDITIONS = ["ON", "OFF"]


# ===========================================================================
# Core routine, reused for the full sample and for every interim sample
# ===========================================================================

def build_participant_table(epoch_table, logger, quiet=False):
    """
    Turn an epoch-level table into a participant-level table.

    Everything that depends on the group (the fit-error z-score) is computed
    inside this function, so calling it with a subset of participants gives
    a correctly rebuilt result for that subset.

    Returns (participant_table, specparam_report, dropped_participants).
    """
    table = epoch_table.copy()

    # Epochs whose FOOOF fit failed outright can never be used.
    # Reading back from CSV can turn the flag into text, so make it a
    # proper boolean first. An epoch with no AUC counts as failed too.
    failed = table["fooof_failed"].astype(str).str.lower().isin(["true", "1"])
    failed = failed | table["auc_2_3"].isna() | table["fit_error"].isna()
    table["fooof_failed"] = failed

    n_failed = int(failed.sum())
    table = table[~failed].copy()

    # ---- 1. z-score the fit error ----------------------------------------
    # Computed separately for the left and right ROI, because those are
    # different spectra, but across all participants within each ROI.
    table["fit_error_z"] = np.nan

    for roi_name in ["left", "right"]:
        is_roi = table["roi"] == roi_name
        errors = table.loc[is_roi, "fit_error"]

        if len(errors) < 2:
            continue

        mean_error = errors.mean()
        sd_error = errors.std(ddof=1)

        if sd_error == 0 or not np.isfinite(sd_error):
            table.loc[is_roi, "fit_error_z"] = 0.0
        else:
            table.loc[is_roi, "fit_error_z"] = (errors - mean_error) / sd_error

        if not quiet:
            logger.info("ROI %-5s : fit error across %d epochs from %d "
                        "participants, mean = %.4f, SD = %.4f",
                        roi_name, len(errors), table.loc[is_roi, "id"].nunique(),
                        mean_error, sd_error)

    table["dropped_specparam"] = table["fit_error_z"] > cfg.FIT_ERROR_Z_THRESHOLD

    if not quiet:
        logger.info("FOOOF fits that failed outright : %d", n_failed)
        logger.info("Epochs dropped by the z > %.1f fit screen : %d of %d",
                    cfg.FIT_ERROR_Z_THRESHOLD,
                    int(table["dropped_specparam"].sum()), len(table))

    # ---- 2. per-participant specparam report -----------------------------
    report_rows = []
    for (pid, session, condition, roi_name), group in table.groupby(
            ["id", "session", "condition", "roi"]):
        retained = group[~group["dropped_specparam"]]
        report_rows.append({
            "id": pid,
            "session": session,
            "condition": condition,
            "roi": roi_name,
            "n_epochs_in": len(group),
            "n_dropped_specparam": int(group["dropped_specparam"].sum()),
            "n_retained": len(retained),
            "mean_r_squared": retained["r_squared"].mean(),
            "sd_r_squared": retained["r_squared"].std(ddof=1),
        })
    specparam_report = pd.DataFrame(report_rows)

    # ---- 3. average the retained epochs ----------------------------------
    retained = table[~table["dropped_specparam"]].copy()

    cell_means = (retained
                  .groupby(["id", "session", "condition", "roi"])["auc_2_3"]
                  .mean()
                  .reset_index())

    # ---- 4. keep only the contralateral ROI ------------------------------
    roi_lookup = (retained[["id", "target_roi", "side_of_stimulator"]]
                  .drop_duplicates(subset=["id"]))
    cell_means = cell_means.merge(roi_lookup, on="id", how="left")
    contralateral = cell_means[cell_means["roi"] == cell_means["target_roi"]].copy()

    # ---- 5. reshape into one row per participant -------------------------
    contralateral["cell"] = (contralateral["session"].str.replace("_task_rest", "",
                                                                 regex=False)
                             + "_" + contralateral["condition"])

    wide = contralateral.pivot(index="id", columns="cell",
                               values="auc_2_3").reset_index()

    required_cells = ["pre_ON", "pre_OFF", "post_ON", "post_OFF"]
    for cell in required_cells:
        if cell not in wide.columns:
            wide[cell] = np.nan

    # A participant needs all four cells to contribute a change score.
    incomplete = wide[wide[required_cells].isna().any(axis=1)]
    dropped_participants = list(incomplete["id"])
    if len(dropped_participants) > 0 and not quiet:
        for pid in dropped_participants:
            logger.warning("EXCLUDED %s -- no epochs left in at least one cell "
                           "after the specparam fit screen", pid)

    wide = wide[~wide["id"].isin(dropped_participants)].copy()

    # ---- 6. change scores -------------------------------------------------
    wide["delta_ON"] = np.abs(wide["post_ON"] - wide["pre_ON"])
    wide["delta_OFF"] = np.abs(wide["post_OFF"] - wide["pre_OFF"])
    wide["diff"] = (wide["delta_ON"]) - (wide["delta_OFF"])

    wide = wide.merge(roi_lookup, on="id", how="left")

    ordered_columns = ["id", "side_of_stimulator", "target_roi",
                       "pre_ON", "post_ON", "pre_OFF", "post_OFF",
                       "delta_ON", "delta_OFF", "diff"]
    wide = wide[ordered_columns].sort_values("id").reset_index(drop=True)

    return wide, specparam_report, dropped_participants


# ===========================================================================
# Enrolment order
# ===========================================================================

def get_enrolment_order(participant_ids, logger):
    """
    Return the participant ids in the order they were enrolled.

    Uses the ORDER_COLUMN of the contralaterality CSV if it is present,
    otherwise falls back to sorting the folder names.
    """
    # Use the same tolerant column matching as the extraction step, so a
    # header with stray spaces or odd capitalisation behaves identically.
    try:
        raw_table = pd.read_csv(cfg.CONTRALATERALITY_CSV)
        order_col = find_column(raw_table.columns, [cfg.ORDER_COLUMN, "order",
                                                    "enrolment", "enrollment"],
                                logger, "order")
        id_col = find_column(raw_table.columns,
                             [cfg.ID_COLUMN, "participant id", "participant_id",
                              "participant", "subject id", "subject", "id"],
                             logger, "participant id")
    except Exception as error:
        logger.warning("Could not read %s (%s). Falling back to folder order.",
                       cfg.CONTRALATERALITY_CSV, error)
        order_col = None
        id_col = None

    if order_col is not None and id_col is not None:
        raw_table[id_col] = raw_table[id_col].astype(str).str.strip()
        raw_table["matched"] = [
            normalise_participant_id(value, participant_ids)
            for value in raw_table[id_col]]
        raw_table = raw_table[raw_table["matched"].notna()]
        raw_table = raw_table.sort_values(order_col)
        logger.info("Enrolment order taken from the '%s' column", order_col)
        return list(raw_table["matched"])

    logger.info("No '%s' column found; enrolment order assumed to follow the "
                "participant numbering (hx_01, hx_02, ...)", cfg.ORDER_COLUMN)
    return sorted(participant_ids)


# ===========================================================================
# Main
# ===========================================================================

def main():
    logger = setup_logging(cfg.OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("H1 group-level assembly started")
    logger.info("=" * 70)

    epoch_path = os.path.join(cfg.OUTPUT_DIR, "h1_epoch_level.csv")
    epoch_table = pd.read_csv(epoch_path)
    logger.info("Read %d epoch-level rows from %d participants",
                len(epoch_table), epoch_table["id"].nunique())

    # ---- full sample ------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Building the participant-level table on the FULL sample")

    participant_table, specparam_report, dropped = build_participant_table(
        epoch_table, logger)

    participant_table.to_csv(
        os.path.join(cfg.OUTPUT_DIR, "h1_participant_level.csv"), index=False)
    specparam_report.to_csv(
        os.path.join(cfg.OUTPUT_DIR, "h1_specparam_report.csv"), index=False)

    logger.info("Participant-level table written: %d valid participants",
                len(participant_table))

    if len(specparam_report) > 0:
        overall_mean = specparam_report["mean_r_squared"].mean()
        overall_sd = specparam_report["mean_r_squared"].std(ddof=1)
        logger.info("Goodness-of-fit across retained epochs: "
                    "r-squared = %.3f +/- %.3f", overall_mean, overall_sd)

    # Append the late exclusions to the exclusion file.
    if len(dropped) > 0:
        exclusion_path = os.path.join(cfg.OUTPUT_DIR, "h1_exclusions.csv")
        existing = pd.read_csv(exclusion_path) if os.path.exists(exclusion_path) \
            else pd.DataFrame(columns=["id", "reason", "detail"])
        new_rows = pd.DataFrame([
            {"id": pid, "reason": "no_epochs_after_specparam", "detail": ""}
            for pid in dropped])
        pd.concat([existing, new_rows], ignore_index=True).to_csv(
            exclusion_path, index=False)

    # ---- interim samples for the early stopping rule ----------------------
    logger.info("-" * 70)
    logger.info("Rebuilding the analysis for every interim sample size")

    valid_ids = list(participant_table["id"])
    ordered_ids = [pid for pid in get_enrolment_order(valid_ids, logger)
                   if pid in valid_ids]

    sequential_rows = []
    n_available = len(ordered_ids)

    if n_available < cfg.SEQUENTIAL_N_MIN:
        logger.warning("Only %d valid participants. The early stopping rule "
                       "does not begin until N = %d, so no interim analysis "
                       "is produced yet.", n_available, cfg.SEQUENTIAL_N_MIN)
    else:
        highest_n = min(n_available, cfg.SEQUENTIAL_N_MAX)
        for n in range(cfg.SEQUENTIAL_N_MIN, highest_n + 1):
            subset_ids = ordered_ids[:n]
            subset_epochs = epoch_table[epoch_table["id"].isin(subset_ids)]

            subset_table, _, subset_dropped = build_participant_table(
                subset_epochs, logger, quiet=True)

            logger.info("Interim N = %2d : %d participants entered the test "
                        "(%d lost at the specparam step on this re-run)",
                        n, len(subset_table), len(subset_dropped))

            for _, row in subset_table.iterrows():
                sequential_rows.append({
                    "N": n,
                    "id": row["id"],
                    "delta_ON": row["delta_ON"],
                    "delta_OFF": row["delta_OFF"],
                    "diff": row["diff"],
                })

    sequential_table = pd.DataFrame(
        sequential_rows, columns=["N", "id", "delta_ON", "delta_OFF", "diff"])
    sequential_table.to_csv(
        os.path.join(cfg.OUTPUT_DIR, "h1_sequential_input.csv"), index=False)

    logger.info("=" * 70)
    logger.info("Group assembly complete")
    logger.info("REMINDER: the early stopping rule requires all FOUR "
                "hypotheses to reach a conclusive Bayes factor. This pipeline "
                "evaluates H1 only, so its output is an interim record and "
                "never a stop decision on its own.")
    logger.info("Next step: run h1_analysis.R")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
