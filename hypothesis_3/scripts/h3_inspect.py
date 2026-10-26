# -*- coding: utf-8 -*-
"""
Inspection for the Hypothesis 3 pipeline.

Checks that the continuous task recordings are epoched the way they should be,
and plots the ERS so the result can be eyeballed before a full run. Nothing
here is used by the extraction script; it only reads and reports.

    python h3_inspect.py
"""

import os

import numpy as np
import pandas as pd
import mne

import h3_config as cfg
from h3_trials import find_trials, count_codes
from h3_ers import (find_participants, load_participant_info, roi_for_participant,
                    build_epochs, percent_change, collapse_trials, to_tfr_image,
                    to_timecourse, trimmed_times, average, output_tag)
from h3_plots import plot_group_tfr, plot_group_timecourse

mne.set_log_level("ERROR")


def inspect_participant(participant, path, roi_channels):
    """Report on one participant. Returns (summary, trial rows, problems)."""
    raw = mne.io.read_raw_fif(path, preload=False)
    sfreq = raw.info["sfreq"]

    trials = find_trials(raw)
    accepted = [t for t in trials if t["status"] == "ok"]

    problems = []
    baseline_epochs, task_epochs, kept = build_epochs(raw, accepted,
                                                      picks=roi_channels)
    n_epochs = 0 if task_epochs is None else len(task_epochs)

    if n_epochs == 0:
        problems.append("no epochs built")
    else:
        # 1. the two epoch sets must be trial-for-trial matched
        if len(baseline_epochs) != len(task_epochs):
            problems.append("%d baseline epochs but %d analysis epochs"
                            % (len(baseline_epochs), len(task_epochs)))

        # 2. every accepted trial should have survived into both sets
        if n_epochs != len(accepted):
            problems.append("%d accepted trials but %d epochs"
                            % (len(accepted), n_epochs))

        # 3. each epoch must be time-locked to its own marker: the baseline set
        #    to movement onset, the analysis set to movement stop
        expected_onset = np.array([accepted[i]["onset_sample"] for i in kept])
        expected_stop = np.array([accepted[i]["stop_sample"] for i in kept])
        if not np.array_equal(baseline_epochs.events[:, 0], expected_onset):
            problems.append("baseline epochs are not locked to the onset markers")
        if not np.array_equal(task_epochs.events[:, 0], expected_stop):
            problems.append("analysis epochs are not locked to the stop markers")

        # 4. both windows must be the ones asked for and must contain the
        #    windows that are actually used
        base_times, task_times = baseline_epochs.times, task_epochs.times
        if not (np.isclose(base_times[0], cfg.BASE_TMIN_PAD, atol=1 / sfreq)
                and np.isclose(base_times[-1], cfg.BASE_TMAX_PAD, atol=1 / sfreq)):
            problems.append("baseline epoch is %.3f to %.3f s, expected %.1f to %.1f s"
                            % (base_times[0], base_times[-1],
                               cfg.BASE_TMIN_PAD, cfg.BASE_TMAX_PAD))
        if not (np.isclose(task_times[0], cfg.TMIN_PAD, atol=1 / sfreq)
                and np.isclose(task_times[-1], cfg.TMAX_PAD, atol=1 / sfreq)):
            problems.append("analysis epoch is %.3f to %.3f s, expected %.1f to %.1f s"
                            % (task_times[0], task_times[-1],
                               cfg.TMIN_PAD, cfg.TMAX_PAD))
        if base_times[0] > cfg.BASELINE[0] or base_times[-1] < cfg.BASELINE[1]:
            problems.append("baseline epoch does not cover the baseline window")
        if task_times[0] > cfg.TMIN or task_times[-1] < cfg.ERS_WINDOW[1]:
            problems.append("analysis epoch does not cover the ERS window")

    # 5. the accepted trials must all satisfy the waiting rule
    waits = [t["wait_s"] for t in accepted]
    if waits and min(waits) < cfg.MIN_WAIT_S:
        problems.append("a trial shorter than %.1f s survived" % cfg.MIN_WAIT_S)

    # trial-by-trial rows for the CSV
    kept_set = set(int(k) for k in kept)
    rows, accepted_index = [], 0
    for trial in trials:
        in_epochs = ""
        if trial["status"] == "ok":
            in_epochs = "yes" if accepted_index in kept_set else "no"
            accepted_index += 1
        rows.append({
            "participant": participant,
            "condition": trial["condition"],
            "direction": trial["direction"],
            "standing_s": round(trial["standing_s"], 3),
            "onset_s": None if trial["onset_s"] is None else round(trial["onset_s"], 3),
            "stop_s": None if trial["stop_s"] is None else round(trial["stop_s"], 3),
            "wait_s": None if trial["wait_s"] is None else round(trial["wait_s"], 3),
            "walk_s": None if trial["walk_s"] is None else round(trial["walk_s"], 3),
            "status": trial["status"],
            "in_epochs": in_epochs,
        })

    walks = [t["walk_s"] for t in accepted]
    summary = {
        "participant": participant,
        "standing": len(trials),
        "unmatched": sum(t["status"] == "unmatched" for t in trials),
        "short": sum(t["status"] == "short_wait" for t in trials),
        "ON": sum(t["condition"] == "ON" for t in accepted),
        "OFF": sum(t["condition"] == "OFF" for t in accepted),
        "epochs": n_epochs,
        "wait_min": round(min(waits), 2) if waits else None,
        "walk_min": round(min(walks), 2) if walks else None,
        "walk_med": round(float(np.median(walks)), 2) if walks else None,
        "walk_max": round(max(walks), 2) if walks else None,
    }
    return summary, rows, problems


def participant_ers(raw, trials, roi_channels):
    """ON and OFF group-TFR and ERS time course for one participant.

    No AutoReject here - this is a quick look at the epoching, not the final
    result. The TFR image is smoothed with PLOT_SMOOTH_MS (lighter, display
    only); the time course uses SMOOTH_MS, the window the real AUC will use.
    """
    result = {}
    for condition in ("ON", "OFF"):
        condition_trials = [t for t in trials
                            if t["condition"] == condition and t["status"] == "ok"]
        if not condition_trials:
            continue
        baseline_epochs, task_epochs, kept = build_epochs(raw, condition_trials,
                                                          picks=roi_channels)
        if baseline_epochs is None:
            continue
        percent, times = percent_change(baseline_epochs, task_epochs)
        by_trial = collapse_trials(percent)
        result[condition] = (to_tfr_image(by_trial, times, cfg.PLOT_SMOOTH_MS),
                             to_timecourse(by_trial, times, cfg.SMOOTH_MS),
                             trimmed_times(times))
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

    if cfg.BASELINE_MODE == "stop":
        print("BASELINE_MODE = 'stop'  --  EXPLORATORY, NOT THE PROTOCOL.")
        print("Baseline is %.0f to %.0f s around movement stop (the end of the walk),"
              % cfg.BASELINE_STOP)
        print("not the pre-gait standing period. The curve starts near zero by")
        print("construction. Do not report these numbers as the H3 result.")
        print()

    header = ("%-10s %8s %9s %6s %5s %5s %7s %8s %8s %8s %8s"
              % ("subject", "standing", "unmatched", "short", "ON", "OFF",
                 "epochs", "wait_min", "walk_min", "walk_med", "walk_max"))
    print(header)
    print("-" * len(header))

    summaries, all_rows, all_problems = [], [], []
    code_totals = {}
    tfr_by_condition = {"ON": [], "OFF": []}
    timecourse_by_condition = {"ON": [], "OFF": []}
    times = None

    for participant, path in participants:
        roi_channels = roi_for_participant(participant_info, participant)
        summary, rows, problems = inspect_participant(participant, path, roi_channels)
        summaries.append(summary)
        all_rows.extend(rows)
        all_problems.extend((participant, p) for p in problems)

        print("%-10s %8d %9d %6d %5d %5d %7d %8s %8s %8s %8s"
              % (summary["participant"], summary["standing"], summary["unmatched"],
                 summary["short"], summary["ON"], summary["OFF"], summary["epochs"],
                 summary["wait_min"], summary["walk_min"], summary["walk_med"],
                 summary["walk_max"]))

        raw = mne.io.read_raw_fif(path, preload=False)
        for code, count in count_codes(raw).items():
            code_totals[code] = code_totals.get(code, 0) + count

        ers = participant_ers(raw, find_trials(raw), roi_channels)
        for condition, (tfr_2d, timecourse, cond_times) in ers.items():
            tfr_by_condition[condition].append(tfr_2d)
            timecourse_by_condition[condition].append(timecourse)
            times = cond_times

    # Marker inventory, so a wrong or missing trigger code is obvious at a glance.
    print()
    print("Target markers found across all files")
    for code in sorted(code_totals):
        print("  %s  %6d" % (code, code_totals[code]))

    print()
    if all_problems:
        print("Problems")
        for participant, problem in all_problems:
            print("  %-10s %s" % (participant, problem))
    else:
        print("No problems: every accepted trial produced one baseline epoch locked "
              "to its movement onset and one analysis epoch locked to its movement "
              "stop, both covering the windows they are used for.")

    below = [s["participant"] for s in summaries
             if min(s["ON"], s["OFF"]) < cfg.MIN_TRIALS]
    if below:
        print()
        print("Below %d trials in a condition: %s" % (cfg.MIN_TRIALS, ", ".join(below)))

    # Tag the outputs with the averaging and baseline used, so an exploratory
    # run cannot silently overwrite the protocol one.
    tag = output_tag()

    out = os.path.join(cfg.OUTPUT_DIR, "h3_inspection_trials.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print()
    print("Trial-by-trial detail: %s" % out)

    if tfr_by_condition["ON"] and tfr_by_condition["OFF"]:
        tfr_on = average(np.stack(tfr_by_condition["ON"]), axis=0)
        tfr_off = average(np.stack(tfr_by_condition["OFF"]), axis=0)
        on_curves = np.stack(timecourse_by_condition["ON"])
        off_curves = np.stack(timecourse_by_condition["OFF"])

        tfr_path = os.path.join(cfg.OUTPUT_DIR, "h3_inspection_tfr%s.png" % tag)
        timecourse_path = os.path.join(
            cfg.OUTPUT_DIR, "h3_inspection_timecourse%s.png" % tag)
        plot_group_tfr(times, cfg.FREQS, tfr_on, tfr_off, tfr_path)
        plot_group_timecourse(times, average(on_curves, axis=0),
                              average(off_curves, axis=0),
                              on_curves, off_curves, timecourse_path)

        print("Group TFR (no AutoReject): %s" % tfr_path)
        print("Group ERS time course (no AutoReject): %s" % timecourse_path)
    else:
        print()
        print("Not enough accepted trials in one condition to plot the group ERS.")


if __name__ == "__main__":
    main()
