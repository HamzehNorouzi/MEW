# -*- coding: utf-8 -*-
"""
Hypothesis 2 — preparatory beta ERD, ON vs OFF.

Reads the stage-0 cleaned task recordings and writes one AUC per participant
per condition. Takes no arguments: everything comes from h2_config.py.

    python h2_extract.py

Writes into OUTPUT_DIR:
    h2_erd.csv                 participant, condition, n_trials, auc
    h2_group_tfr.npz           per-participant TFRs and ERD curves, for plotting
    h2_group_tfr.png           group beta TFR, ON and OFF
    h2_group_timecourse.png    group ERD time course, ON vs OFF
    h2_run.log
"""

import os

# Keep BLAS single-threaded: participants already run in parallel, and the two
# levels of threading otherwise fight over cores. Must precede numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from autoreject import AutoReject

import h2_config_px_hx as cfg
from h2_trials import find_trials, make_events
from h2_erd import (find_participants, load_participant_info, roi_for_participant,
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


def count_by_condition(events):
    """Trials per condition in an events array."""
    return {condition: int(np.sum(events[:, 2] == code))
            for condition, code in cfg.CONDITION_IDS.items()}


def process_participant(participant, path, roi_channels):
    """Everything for one participant. Returns a result dict; `excluded` is a
    reason string when the participant drops out, otherwise None."""
    result = {"participant": participant, "excluded": None, "rows": [],
              "tfr": {}, "timecourse": {}, "times": None, "message": ""}

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
        if min(after.values()) < cfg.MIN_TRIALS:
            result["excluded"] = ("%d ON and %d OFF trials after AutoReject"
                                  % (after["ON"], after["OFF"]))
            return result

        # The ERD only ever comes from the ROI, so the TFR is computed there only.
        epochs.pick(roi_channels)
        percent, times = percent_change(epochs)

        kept_times = trimmed_times(times)
        window = ((kept_times >= cfg.ERD_WINDOW[0] - 1e-6)
                  & (kept_times <= cfg.ERD_WINDOW[1] + 1e-6))

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

    participants = find_participants()
    participant_info = load_participant_info()
    jobs = [(participant, path, roi_for_participant(participant_info, participant))
            for participant, path in participants]

    results = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(process_participant)(*job) for job in jobs)

    rows, included = [], []
    for result in results:
        if result["excluded"]:
            log.info("%-8s excluded: %s", result["participant"], result["excluded"])
            continue
        log.info(result["message"])
        rows.extend(result["rows"])
        included.append(result)

    if not rows:
        log.info("No participants survived.")
        return

    # A non-default averaging setting tags every output, so a comparison run
    # cannot silently replace the results the statistics are based on.
    tag = output_tag()
    if tag:
        log.info("AVERAGING = '%s' -- outputs tagged '%s', not the project default",
                 cfg.AVERAGING, tag)

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
