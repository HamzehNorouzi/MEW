# -*- coding: utf-8 -*-
"""
Hypothesis 2 — preparatory beta ERD, ON vs OFF.

Reads the stage-0 cleaned task recordings and writes one AUC per participant
per condition. Everything else comes from h2_config.py.

On startup it asks which cohort to load, hx_XX or px_XX. To skip the question
in an unattended run, pass the cohort on the command line or set H2_COHORT:

    python h2_extract.py
    python h2_extract.py px
    H2_COHORT=px python h2_extract.py

The cohort decides the minimum stand-up wait, the epoch/baseline/ERD windows
and whether the minimum-trial exclusion applies — see h2_config.py.

Writes into OUTPUT_DIR (px_ outputs carry a "_px" tag so they cannot overwrite
the hx_ results):
    h2_erd.csv                 participant, condition, n_trials, auc
    h2_group_tfr.npz           per-participant TFRs and ERD curves, for plotting
    h2_group_tfr.png           group beta TFR, ON and OFF
    h2_group_timecourse.png    group ERD time course, ON vs OFF
    h2_run.log
"""

import os
import sys

# Keep BLAS single-threaded: participants already run in parallel, and the two
# levels of threading otherwise fight over cores. Must precede numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ---- Which cohort to load ----------------------------------------------------
# Asked BEFORE h2_config is imported, because the cohort decides MIN_WAIT_S,
# MIN_TRIALS and the epoch/baseline/ERD windows that module defines.
#
# The answer goes into the environment rather than into a module variable
# because this script runs participants through joblib, and the worker
# processes import h2_config from scratch. A module attribute set after import
# would not reach them: the parent would use the px_ timing while every worker
# silently used the hx_ timing. Environment variables are inherited.

VALID_COHORTS = ("hx", "px")


def choose_cohort(argv=None):
    """
    Ask whether to load hx_XX or px_XX folders, and return "hx" or "px".

    A cohort given on the command line (``python h2_extract.py px``) or already
    present in H2_COHORT is used without asking, so the script can run
    unattended. Otherwise the question is repeated until the answer is valid,
    because the wrong cohort here silently analyses nothing.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)

    preset = None
    if arguments:
        preset = ("command line", arguments[0])
    elif os.environ.get("H2_COHORT"):
        preset = ("H2_COHORT", os.environ["H2_COHORT"])

    if preset is not None:
        source, value = preset
        token = str(value).strip().lower().rstrip("_")
        if token in VALID_COHORTS:
            return token
        raise SystemExit("Unrecognised cohort %r from the %s. Use one of: %s"
                         % (value, source, ", ".join(VALID_COHORTS)))

    while True:
        try:
            answer = input("Which files should be loaded? [hx/px]: ")
        except EOFError:
            raise SystemExit(
                "No cohort given and no console to ask on. Pass it on the "
                "command line instead, e.g.  python h2_extract.py hx")
        token = answer.strip().lower().rstrip("_")
        if token in VALID_COHORTS:
            return token
        print("Please type 'hx' or 'px'.")


os.environ["H2_COHORT"] = choose_cohort()


import glob
import logging

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from autoreject import AutoReject

import h2_config_px_hx as cfg
from h2_trials import find_trials, make_events
# NOTE find_participants is deliberately NOT imported from h2_erd: that version
# is hardcoded to hx_* and would ignore the cohort chosen above. The
# cohort-aware replacement lives below.
from h2_erd import (load_participant_info, roi_for_participant,
                    percent_change, collapse_trials, to_tfr_image, to_timecourse,
                    trimmed_times, average, output_tag)
from h2_plots import plot_group_tfr, plot_group_timecourse

mne.set_log_level("ERROR")

# numpy renamed trapz to trapezoid in 2.0 and removed the old name
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

log = logging.getLogger("h2")


def setup_logging():
    logfile = os.path.join(cfg.OUTPUT_DIR, "h2_run.log")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    for handler in (logging.FileHandler(logfile, mode="a", encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


def find_participants():
    """
    Find the cleaned task recordings for the cohort chosen at startup.

    Returns a list of (participant, path) pairs, sorted by participant id, for
    the expected stage-0 layout:

        DERIVATIVES_DIR/<prefix>XX/<prefix>XX_task_clean-raw.fif

    If that exact filename is absent, any *task*-raw.fif in the folder is
    accepted instead, so a recording saved under a slightly different name is
    still picked up rather than silently skipped. Folders with no task file at
    all are reported by the caller, not dropped in silence.
    """
    found, without_task = [], []

    for folder in sorted(glob.glob(os.path.join(cfg.DERIVATIVES_DIR,
                                                cfg.SUBJECT_PREFIX + "*"))):
        if not os.path.isdir(folder):
            continue

        participant = os.path.basename(folder)
        expected = os.path.join(folder, "%s_task_clean-raw.fif" % participant)

        if os.path.exists(expected):
            found.append((participant, expected))
            continue

        # "task" also appears inside "pre_task_rest" / "post_task_rest", so the
        # rest stems are excluded explicitly. Without this a participant whose
        # task file is missing would silently have a RESTING-STATE recording
        # analysed as though it were the task.
        alternatives = sorted(
            path for path in glob.glob(os.path.join(folder, "*task*-raw.fif"))
            if os.path.isfile(path) and "rest" not in os.path.basename(path).lower()
        )
        if alternatives:
            found.append((participant, alternatives[0]))
        else:
            without_task.append(participant)

    return found, without_task


def count_by_condition(events):
    """Trials per condition in an events array."""
    return {condition: int(np.sum(events[:, 2] == code))
            for condition, code in cfg.CONDITION_IDS.items()}


def process_participant(participant, path, roi_channels):
    """Everything for one participant. Returns a result dict; `excluded` is a
    reason string when the participant drops out, otherwise None."""
    result = {"participant": participant, "excluded": None, "warning": None,
              "rows": [], "tfr": {}, "timecourse": {}, "times": None,
              "message": ""}

    try:
        raw = mne.io.read_raw_fif(path, preload=False)
        trials = [t for t in find_trials(raw) if t["status"] == "ok"]
        events = make_events(trials)

        if len(events) == 0:
            result["excluded"] = "no valid trials"
            return result

        # All channels here: AutoReject is fitted on the whole head.
        epochs = mne.Epochs(raw, events, event_id=cfg.CONDITION_IDS,
                            tmin=cfg.TMIN_PAD, tmax=cfg.TMAX_PAD,
                            baseline=None, preload=True, verbose="ERROR")
        before = count_by_condition(epochs.events)

        # AutoReject judges the analysis window, both conditions pooled, so the
        # threshold cannot depend on condition. Only its drop log is used.
        core = epochs.copy().crop(cfg.TMIN, cfg.TMAX)
        ar = AutoReject(n_interpolate=cfg.AR_N_INTERPOLATE,
                        consensus=cfg.AR_CONSENSUS, cv=cfg.AR_CV,
                        random_state=cfg.RANDOM_STATE, n_jobs=1, verbose=False)
        ar.fit(core)
        keep = np.where(~ar.get_reject_log(core).bad_epochs)[0]
        epochs = epochs[keep]

        after = count_by_condition(epochs.events)

        # The minimum-trial rule. h2_config sets MIN_TRIALS = 0 for the px_
        # cohort, so a px_ participant is never dropped here however few trials
        # survive; for hx_ the usual 30-trial rule applies.
        if min(after.values()) < cfg.MIN_TRIALS:
            result["excluded"] = ("%d ON and %d OFF trials after AutoReject "
                                  "(minimum %d)"
                                  % (after["ON"], after["OFF"], cfg.MIN_TRIALS))
            return result

        # Keeping a participant is not the same as being able to measure one.
        # With no trials at all in a condition there is no spectrum to collapse
        # and no AUC to integrate, so this is reported and no row is written.
        if min(after.values()) == 0:
            result["excluded"] = ("no trials left in one condition "
                                  "(ON %d, OFF %d) — no AUC computable"
                                  % (after["ON"], after["OFF"]))
            return result

        # "Never dropped" must not become "never noticed".
        if min(after.values()) < cfg.LOW_TRIAL_WARNING:
            result["warning"] = ("kept with only %d ON / %d OFF trials, below "
                                 "the usual %d-trial rule (%s cohort)"
                                 % (after["ON"], after["OFF"],
                                    cfg.LOW_TRIAL_WARNING, cfg.SUBJECT_PREFIX))

        # The ERD only ever comes from the ROI, so the TFR is computed there only.
        epochs.pick(roi_channels)
        percent, times = percent_change(epochs)

        kept_times = trimmed_times(times)
        window = ((kept_times >= cfg.ERD_WINDOW[0] - 1e-6)
                  & (kept_times <= cfg.ERD_WINDOW[1] + 1e-6))

        # A window that selects nothing means the ERD window and the epoch no
        # longer line up — a cohort/config mismatch rather than bad data.
        if not window.any():
            result["excluded"] = (
                "ERD_WINDOW %s selects no samples from the %.1f..%.1f s epoch"
                % (cfg.ERD_WINDOW, kept_times[0], kept_times[-1]))
            return result

        for condition, code in cfg.CONDITION_IDS.items():
            mask = epochs.events[:, 2] == code
            by_trial = collapse_trials(percent[mask])

            # SMOOTH_MS shapes the curve the AUC integrates: the locked analysis
            # parameter. PLOT_SMOOTH_MS is lighter and only touches the TFR image.
            timecourse = to_timecourse(by_trial, times, cfg.SMOOTH_MS)
            tfr_2d = to_tfr_image(by_trial, times, cfg.PLOT_SMOOTH_MS)

            auc = float(_trapz(timecourse[window], kept_times[window]))

            result["rows"].append({"participant": participant,
                                   "condition": condition,
                                   "n_trials": int(mask.sum()),
                                   "auc": auc})
            result["tfr"][condition] = tfr_2d
            result["timecourse"][condition] = timecourse
            result["times"] = kept_times

        result["message"] = ("%-8s ON %d/%d   OFF %d/%d   AutoReject dropped %d"
                             % (participant, after["ON"], before["ON"],
                                after["OFF"], before["OFF"],
                                len(events) - len(keep)))
    except Exception as error:
        result["excluded"] = "%s: %s" % (type(error).__name__, error)

    return result


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logging()

    log.info("=" * 70)
    log.info("Cohort: %sXX   (MIN_WAIT_S = %s, MIN_TRIALS = %s, "
             "baseline %s, ERD window %s)",
             cfg.SUBJECT_PREFIX, cfg.MIN_WAIT_S, cfg.MIN_TRIALS,
             cfg.BASELINE, cfg.ERD_WINDOW)
    log.info("=" * 70)

    participants, without_task = find_participants()

    if without_task:
        log.info("%d %sXX folder(s) have no task recording, skipped: %s",
                 len(without_task), cfg.SUBJECT_PREFIX,
                 ", ".join(without_task))

    if not participants:
        available = sorted(
            os.path.basename(f)
            for f in glob.glob(os.path.join(cfg.DERIVATIVES_DIR, "*"))
            if os.path.isdir(f)
        )
        raise SystemExit(
            "No %sXX participant folders with a task recording in %s.\n"
            "Folders that ARE there: %s\n"
            "Re-run and choose the other cohort if that is what you meant."
            % (cfg.SUBJECT_PREFIX, cfg.DERIVATIVES_DIR,
               ", ".join(available[:12]) + (" ..." if len(available) > 12 else "")))

    log.info("Found %d %sXX participants", len(participants), cfg.SUBJECT_PREFIX)

    participant_info = load_participant_info()

    # The other place a hardcoded hx_ assumption can hide is the participant-id
    # matching against participants_info.csv. If a px_ id does not match, this
    # either raises or hands back an empty ROI, and every participant would then
    # fail deep inside a worker with an obscure message. Checked here instead.
    jobs = []
    for participant, path in participants:
        try:
            roi = roi_for_participant(participant_info, participant)
        except Exception as error:
            raise SystemExit(
                "roi_for_participant() failed for %s: %s: %s\n"
                "The id in %s probably does not match the %sXX folder names. "
                "Check how h2_erd.py matches ids — the same hardcoded-prefix "
                "problem as find_participants()."
                % (participant, type(error).__name__, error,
                   cfg.PARTICIPANT_INFO, cfg.SUBJECT_PREFIX))

        if roi is None or len(roi) == 0:
            raise SystemExit(
                "roi_for_participant() returned no ROI channels for %s.\n"
                "Either %s has no row for this participant, or its id does not "
                "match the %sXX folder naming. Check h2_erd.py."
                % (participant, cfg.PARTICIPANT_INFO, cfg.SUBJECT_PREFIX))

        jobs.append((participant, path, roi))

    results = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(process_participant)(*job) for job in jobs)

    rows, included = [], []
    for result in results:
        if result["excluded"]:
            log.info("%-8s excluded: %s", result["participant"], result["excluded"])
            continue
        log.info(result["message"])
        if result["warning"]:
            log.info("%-8s WARNING: %s", result["participant"], result["warning"])
        rows.extend(result["rows"])
        included.append(result)

    if not rows:
        log.info("No participants survived.")
        return

    # A non-default averaging setting tags every output, so a comparison run
    # cannot silently replace the results the statistics are based on. The
    # cohort tag does the same job across cohorts: px_ outputs never overwrite
    # the hx_ ones.
    tag = cfg.COHORT_TAG + output_tag()
    if output_tag():
        log.info("AVERAGING = '%s' -- outputs tagged '%s', not the project default",
                 cfg.AVERAGING, output_tag())
    if cfg.COHORT_TAG:
        log.info("Cohort '%s' -- outputs tagged '%s'", cfg.COHORT, cfg.COHORT_TAG)

    erd_csv = os.path.join(cfg.OUTPUT_DIR, "h2_erd%s.csv" % tag)
    pd.DataFrame(rows).to_csv(erd_csv, index=False)

    # Stacked as (participant, condition, ...) with conditions in a fixed order.
    conditions = list(cfg.CONDITION_IDS)
    times = included[0]["times"]
    tfr = np.stack([[r["tfr"][c] for c in conditions] for r in included])
    timecourse = np.stack([[r["timecourse"][c] for c in conditions] for r in included])

    np.savez_compressed(os.path.join(cfg.OUTPUT_DIR, "h2_group_tfr%s.npz" % tag),
                        participants=np.array([r["participant"] for r in included]),
                        conditions=np.array(conditions), times=times, freqs=cfg.FREQS,
                        tfr=tfr, timecourse=timecourse)

    on, off = conditions.index("ON"), conditions.index("OFF")
    plot_group_tfr(times, cfg.FREQS,
                   average(tfr[:, on], axis=0), average(tfr[:, off], axis=0),
                   os.path.join(cfg.OUTPUT_DIR, "h2_group_tfr%s.png" % tag))
    plot_group_timecourse(times,
                          average(timecourse[:, on], axis=0),
                          average(timecourse[:, off], axis=0),
                          timecourse[:, on], timecourse[:, off],
                          os.path.join(cfg.OUTPUT_DIR, "h2_group_timecourse%s.png" % tag))

    log.info("%d participants -> %s", len(included), erd_csv)


if __name__ == "__main__":
    main()
