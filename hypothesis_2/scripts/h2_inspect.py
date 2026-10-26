# -*- coding: utf-8 -*-
"""
Inspection for the Hypothesis 2 pipeline.

Checks that the continuous task recordings are epoched the way they should be.
Nothing here is used by the extraction script; it only reads and reports.

    python h2_inspect.py
"""

import os

import numpy as np
import pandas as pd
import mne

import h2_config_px_hx as cfg
from h2_trials import find_trials, make_events
from h2_erd import (find_participants, load_participant_info, roi_for_participant,
                    condition_percent_change, collapse_trials, to_tfr_image,
                    to_timecourse, trimmed_times, average, output_tag)
from h2_plots import plot_group_tfr, plot_group_timecourse

mne.set_log_level("ERROR")


def inspect_participant(participant, path):
    """Report on one participant. Returns (summary dict, trial rows, problems)."""
    raw = mne.io.read_raw_fif(path, preload=False)
    sfreq = raw.info["sfreq"]

    trials = find_trials(raw)
    accepted = [t for t in trials if t["status"] == "ok"]
    events = make_events(accepted)

    problems = []
    n_epochs = 0
    epochs = None

    if len(events) == 0:
        problems.append("no accepted trials")
    else:
        epochs = mne.Epochs(raw, events, event_id=cfg.CONDITION_IDS,
                            tmin=cfg.TMIN_PAD, tmax=cfg.TMAX_PAD,
                            baseline=None, preload=False)
        epochs.drop_bad()          # so selection and length are final
        n_epochs = len(epochs)

        # 1. every accepted trial should have become an epoch
        if n_epochs != len(accepted):
            problems.append(
                "%d accepted trials but %d epochs" % (len(accepted), n_epochs))

        # 2. each epoch must be time-locked to its own movement-onset marker
        kept_samples = epochs.events[:, 0]
        expected = np.array([t["onset_sample"] for t in accepted])[epochs.selection]
        if not np.array_equal(kept_samples, expected):
            problems.append("epoch events do not match the onset markers")

        # 3. the epoch window must be the one asked for, and must contain both
        #    the baseline and the ERD window
        times = epochs.times
        if not (np.isclose(times[0], cfg.TMIN_PAD, atol=1 / sfreq)
                and np.isclose(times[-1], cfg.TMAX_PAD, atol=1 / sfreq)):
            problems.append("epoch window is %.3f to %.3f s, expected %.1f to %.1f s"
                            % (times[0], times[-1], cfg.TMIN_PAD, cfg.TMAX_PAD))
        if times[0] > cfg.BASELINE[0] or times[-1] < cfg.ERD_WINDOW[1]:
            problems.append("epoch does not cover the baseline and ERD windows")

    # 4. the accepted trials must all satisfy the waiting rule
    waits = [t["wait_s"] for t in accepted]
    if waits and min(waits) < cfg.MIN_WAIT_S:
        problems.append("a trial shorter than %.1f s survived" % cfg.MIN_WAIT_S)

    # trial-by-trial rows for the CSV
    rows = []
    kept_index = 0
    for trial in trials:
        in_epochs = ""
        if trial["status"] == "ok":
            in_epochs = "yes" if (epochs is not None
                                  and kept_index in epochs.selection) else "no"
            kept_index += 1
        rows.append({
            "participant": participant,
            "condition": trial["condition"],
            "direction": trial["direction"],
            "standup_s": round(trial["standup_s"], 3),
            "onset_s": None if trial["onset_s"] is None else round(trial["onset_s"], 3),
            "wait_s": None if trial["wait_s"] is None else round(trial["wait_s"], 3),
            "status": trial["status"],
            "in_epochs": in_epochs,
        })

    summary = {
        "participant": participant,
        "standups": len(trials),
        "unpaired": sum(t["status"] == "unpaired" for t in trials),
        "short": sum(t["status"] == "short_wait" for t in trials),
        "ON": sum(t["condition"] == "ON" for t in accepted),
        "OFF": sum(t["condition"] == "OFF" for t in accepted),
        "outward": sum(t["direction"] == "outward" for t in accepted),
        "return": sum(t["direction"] == "return" for t in accepted),
        "epochs": n_epochs,
        "wait_min": round(min(waits), 2) if waits else None,
    }
    return summary, rows, problems


def participant_erd(raw, trials, roi_channels):
    """ON and OFF group-TFR and ERD time course for one participant.

    No AutoReject here — this is a quick look at the epoching, not the final
    result. The TFR image is smoothed with PLOT_SMOOTH_MS (lighter, display
    only); the time course is smoothed with SMOOTH_MS, the same window the
    real AUC uses, so this preview matches what h2_extract.py will integrate.
    Returns {"ON": (tfr_2d, timecourse, times), "OFF": (...)}, with a condition
    missing if it had no accepted trials.
    """
    result = {}
    for condition in ("ON", "OFF"):
        condition_trials = [t for t in trials
                            if t["condition"] == condition and t["status"] == "ok"]
        percent, times = condition_percent_change(raw, condition_trials, roi_channels)
        if percent is None:
            continue
        by_trial = collapse_trials(percent)
        tfr_2d = to_tfr_image(by_trial, times, cfg.PLOT_SMOOTH_MS)
        timecourse = to_timecourse(by_trial, times, cfg.SMOOTH_MS)
        result[condition] = (tfr_2d, timecourse, trimmed_times(times))
    return result


def main():
    participants = find_participants()
    if not participants:
        print("No task files found under %s" % cfg.DERIVATIVES_DIR)
        return

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    participant_info = load_participant_info()

    if cfg.AVERAGING != "median":
        print("AVERAGING = '%s', not the project default 'median'." % cfg.AVERAGING)
        print("Outputs are tagged '_mean' and are for comparison only.")
        print()

    header = ("%-10s %8s %8s %6s %5s %5s %8s %7s %8s %8s"
              % ("subject", "standups", "unpaired", "short", "ON", "OFF",
                 "outward", "return", "epochs", "wait_min"))
    print(header)
    print("-" * len(header))

    summaries, all_rows, all_problems = [], [], []
    tfr_by_condition = {"ON": [], "OFF": []}
    timecourse_by_condition = {"ON": [], "OFF": []}
    times = None

    for participant, path in participants:
        summary, rows, problems = inspect_participant(participant, path)
        summaries.append(summary)
        all_rows.extend(rows)
        all_problems.extend((participant, p) for p in problems)

        print("%-10s %8d %8d %6d %5d %5d %8d %7d %8d %8s"
              % (summary["participant"], summary["standups"], summary["unpaired"],
                 summary["short"], summary["ON"], summary["OFF"],
                 summary["outward"], summary["return"], summary["epochs"],
                 summary["wait_min"]))

        roi_channels = roi_for_participant(participant_info, participant)
        raw = mne.io.read_raw_fif(path, preload=False)
        erd = participant_erd(raw, find_trials(raw), roi_channels)
        for condition, (tfr_2d, timecourse, cond_times) in erd.items():
            tfr_by_condition[condition].append(tfr_2d)
            timecourse_by_condition[condition].append(timecourse)
            times = cond_times

    print()
    if all_problems:
        print("Problems")
        for participant, problem in all_problems:
            print("  %-10s %s" % (participant, problem))
    else:
        print("No problems: every accepted trial became one epoch, time-locked to "
              "its own movement-onset marker, over the full %.1f to %.1f s window."
              % (cfg.TMIN_PAD, cfg.TMAX_PAD))

    below = [s["participant"] for s in summaries
             if min(s["ON"], s["OFF"]) < cfg.MIN_TRIALS]
    if below:
        print()
        print("Below %d trials in a condition: %s" % (cfg.MIN_TRIALS, ", ".join(below)))

    out = os.path.join(cfg.OUTPUT_DIR, "h2_inspection_trials.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print()
    print("Trial-by-trial detail: %s" % out)

    if tfr_by_condition["ON"] and tfr_by_condition["OFF"]:
        tag = output_tag()
        tfr_on = average(np.stack(tfr_by_condition["ON"]), axis=0)
        tfr_off = average(np.stack(tfr_by_condition["OFF"]), axis=0)
        on_curves = np.stack(timecourse_by_condition["ON"])
        off_curves = np.stack(timecourse_by_condition["OFF"])

        tfr_path = os.path.join(cfg.OUTPUT_DIR, "h2_inspection_tfr%s.png" % tag)
        timecourse_path = os.path.join(
            cfg.OUTPUT_DIR, "h2_inspection_timecourse%s.png" % tag)
        plot_group_tfr(times, cfg.FREQS, tfr_on, tfr_off, tfr_path)
        plot_group_timecourse(times, average(on_curves, axis=0),
                              average(off_curves, axis=0),
                              on_curves, off_curves, timecourse_path)

        print("Group TFR (no AutoReject): %s" % tfr_path)
        print("Group ERD time course (no AutoReject): %s" % timecourse_path)
    else:
        print()
        print("Not enough accepted trials in one condition to plot the group ERD.")


if __name__ == "__main__":
    main()
