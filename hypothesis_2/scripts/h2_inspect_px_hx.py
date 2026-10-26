# -*- coding: utf-8 -*-
"""
Inspection for the Hypothesis 2 pipeline.

Checks that the continuous task recordings are epoched the way they should be.
Nothing here is used by the extraction script; it only reads and reports.

This script never excludes a trial or a file for a short stand-up-to-onset
wait: it reports every "ok" and "short_wait" trial (only "unpaired" -- no
onset marker at all -- is left out, since there is nothing to epoch). The
MIN_WAIT_S cutoff still applies downstream, in h2_extract.py, which is what
actually builds the AUCs the statistics use; this script exists to let you
see everything first, including the trials that cutoff would drop.

On startup it asks which cohort to inspect, hx_XX or px_XX. To skip the
question, pass the cohort on the command line or set H2_COHORT:

    python h2_inspect.py
    python h2_inspect.py px
    H2_COHORT=px python h2_inspect.py

The cohort decides the minimum stand-up wait and the epoch / baseline / ERD
windows every check below is made against, so it has to be chosen before
h2_config_px_hx is imported.
"""

import os
import sys

# ---- Which cohort to inspect -------------------------------------------------
# Asked BEFORE the config is imported, because the cohort decides MIN_WAIT_S and
# the epoch, baseline and ERD windows that module defines. The answer goes into
# the environment, which is also what h2_extract.py does, so inspecting and
# extracting the same cohort cannot drift apart.

VALID_COHORTS = ("hx", "px")


def choose_cohort(argv=None):
    """
    Ask whether to inspect hx_XX or px_XX folders, and return "hx" or "px".

    A cohort given on the command line (``python h2_inspect.py px``) or already
    present in H2_COHORT is used without asking. Otherwise the question is
    repeated until the answer is valid, because the wrong cohort here inspects
    nothing and reports no problems, which looks like a clean result.
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
            answer = input("Which files should be inspected? [hx/px]: ")
        except EOFError:
            raise SystemExit(
                "No cohort given and no console to ask on. Pass it on the "
                "command line instead, e.g.  python h2_inspect.py hx")
        token = answer.strip().lower().rstrip("_")
        if token in VALID_COHORTS:
            return token
        print("Please type 'hx' or 'px'.")


os.environ["H2_COHORT"] = choose_cohort()


import glob

import numpy as np
import pandas as pd
import mne

import h2_config_px_hx as cfg
from h2_trials import find_trials, make_events
# NOTE find_participants is deliberately NOT imported from h2_erd: that version
# is hardcoded to hx_* and would ignore the cohort chosen above. The
# cohort-aware replacement lives below, and matches h2_extract.py's.
from h2_erd import (load_participant_info, roi_for_participant,
                    condition_percent_change, collapse_trials, to_tfr_image,
                    to_timecourse, trimmed_times, average, output_tag)
from h2_plots import plot_group_tfr, plot_group_timecourse

mne.set_log_level("ERROR")


def find_participants():
    """
    Find the cleaned task recordings for the cohort chosen at startup.

    Returns (participants, without_task): a list of (participant, path) pairs
    sorted by participant id, and the ids of folders that hold no task
    recording at all. The expected stage-0 layout is

        DERIVATIVES_DIR/<prefix>XX/<prefix>XX_task_clean-raw.fif

    and any *task*-raw.fif in the folder is accepted as a fallback. "task" also
    appears inside "pre_task_rest" and "post_task_rest", so the rest stems are
    excluded explicitly -- without that a participant whose task file is
    missing would have a RESTING-STATE recording inspected as the task.
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

        alternatives = sorted(
            path for path in glob.glob(os.path.join(folder, "*task*-raw.fif"))
            if os.path.isfile(path) and "rest" not in os.path.basename(path).lower()
        )
        if alternatives:
            found.append((participant, alternatives[0]))
        else:
            without_task.append(participant)

    return found, without_task


#: Statuses this inspection script treats as usable. find_trials() (h2_trials.py)
#: assigns "short_wait" to a trial whose stand-up-to-onset interval is below
#: cfg.MIN_WAIT_S; h2_extract.py drops those from the real analysis, but this
#: script's whole job is to show you what is actually in a recording, so a
#: short-wait trial is epoched and counted here like any other. "unpaired"
#: (no matching movement-onset marker at all) is the only status still left
#: out -- there is no onset sample to build an epoch around, so inspecting it
#: isn't a wait-time choice, it's a trial with no usable event.
INCLUDED_STATUSES = ("ok", "short_wait")


def inspect_participant(participant, path):
    """Report on one participant. Returns (summary dict, trial rows, problems).

    Anything that makes a participant unusable is reported as a problem and the
    inspection moves on: a file that will not load, a recording with only one
    condition, or epoching that fails. An inspection run must survive a bad
    participant, otherwise one unreadable file hides the state of all the rest.

    Trials are included here regardless of the MIN_WAIT_S wait-time rule (see
    INCLUDED_STATUSES above) -- this script reports every usable trial, short
    wait or not, and never excludes a file for it. h2_extract.py still applies
    MIN_WAIT_S when it builds the actual AUCs that feed the statistics.
    """
    problems = []

    try:
        raw = mne.io.read_raw_fif(path, preload=False)
        sfreq = raw.info["sfreq"]
        trials = find_trials(raw)
    except Exception as error:
        problems.append("could not be loaded (%s: %s)"
                        % (type(error).__name__, error))
        return None, [], problems

    accepted = [t for t in trials if t["status"] in INCLUDED_STATUSES]
    events = make_events(accepted)

    n_epochs = 0
    epochs = None

    if len(events) == 0:
        problems.append("no accepted trials")
    else:
        # mne.Epochs raises "No matching events found for OFF (event id 1)" if
        # a named event_id matches nothing. Only the conditions actually
        # present are requested, so a one-condition recording is reported
        # rather than ending the run.
        present = {name: code for name, code in cfg.CONDITION_IDS.items()
                   if np.any(events[:, 2] == code)}
        absent = [name for name in cfg.CONDITION_IDS if name not in present]
        if absent:
            problems.append("no %s trials in the recording — h2_extract.py will "
                            "skip this participant (a paired AUC needs both "
                            "conditions)" % " or ".join(absent))

        try:
            epochs = mne.Epochs(raw, events, event_id=present,
                                tmin=cfg.TMIN_PAD, tmax=cfg.TMAX_PAD,
                                baseline=None, preload=False)
            epochs.drop_bad()      # so selection and length are final
            n_epochs = len(epochs)
        except Exception as error:
            problems.append("epoching failed (%s: %s)"
                            % (type(error).__name__, error))
            epochs = None

    # The epoch checks below only mean anything if epoching actually produced
    # something; when it did not, the problem is already recorded above.
    if epochs is not None:
        # 1. every accepted trial should have become an epoch. A missing
        #    condition is reported separately, so only count the trials that
        #    could have become epochs in the first place.
        expected_epochs = int(np.sum(np.isin(events[:, 2],
                                             list(present.values()))))
        if n_epochs != expected_epochs:
            problems.append(
                "%d epochable trials but %d epochs" % (expected_epochs, n_epochs))

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

    # wait_min is reported for every accepted trial, short-wait included -- this
    # script no longer flags a short wait as a problem (that is a deliberate
    # choice, not a bug), it just tells you the shortest wait that got in.
    waits = [t["wait_s"] for t in accepted]

    # trial-by-trial rows for the CSV
    rows = []
    kept_index = 0
    for trial in trials:
        in_epochs = ""
        if trial["status"] in INCLUDED_STATUSES:
            in_epochs = "yes" if (epochs is not None
                                  and kept_index in epochs.selection) else "no"
            kept_index += 1
        rows.append({
            "cohort": cfg.COHORT,
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
        # Informational only now: short-wait trials are counted here but are
        # NOT excluded -- they are already inside `accepted` (see
        # INCLUDED_STATUSES) and are reflected in ON/OFF/outward/return/epochs
        # below like any other trial.
        "short": sum(t["status"] == "short_wait" for t in trials),
        "ON": sum(t["condition"] == "ON" for t in accepted),
        "OFF": sum(t["condition"] == "OFF" for t in accepted),
        "outward": sum(t["direction"] == "outward" for t in accepted),
        "return": sum(t["direction"] == "return" for t in accepted),
        "epochs": n_epochs,
        "wait_min": round(min(waits), 2) if waits else None,
        "wait_median": round(float(np.median(waits)), 2) if waits else None,
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
                            if t["condition"] == condition
                            and t["status"] in INCLUDED_STATUSES]
        percent, times = condition_percent_change(raw, condition_trials, roi_channels)
        if percent is None:
            continue
        by_trial = collapse_trials(percent)
        tfr_2d = to_tfr_image(by_trial, times, cfg.PLOT_SMOOTH_MS)
        timecourse = to_timecourse(by_trial, times, cfg.SMOOTH_MS)
        result[condition] = (tfr_2d, timecourse, trimmed_times(times))
    return result


def main():
    participants, without_task = find_participants()

    print("Cohort: %sXX   (MIN_WAIT_S = %s, baseline %s, ERD window %s, "
          "epochs %.1f to %.1f s)"
          % (cfg.SUBJECT_PREFIX, cfg.MIN_WAIT_S, cfg.BASELINE, cfg.ERD_WINDOW,
             cfg.TMIN_PAD, cfg.TMAX_PAD))
    print("This inspection does not exclude short-wait trials (MIN_WAIT_S "
          "above is not enforced here) -- 'short' below is a count, not an "
          "exclusion. h2_extract.py still enforces it for the real analysis.")
    print()

    if without_task:
        print("%d %sXX folder(s) have no task recording, skipped: %s"
              % (len(without_task), cfg.SUBJECT_PREFIX, ", ".join(without_task)))
        print()

    if not participants:
        available = sorted(
            os.path.basename(f)
            for f in glob.glob(os.path.join(cfg.DERIVATIVES_DIR, "*"))
            if os.path.isdir(f)
        )
        print("No %sXX task files found under %s"
              % (cfg.SUBJECT_PREFIX, cfg.DERIVATIVES_DIR))
        if available:
            print("Folders that ARE there: %s"
                  % (", ".join(available[:12])
                     + (" ..." if len(available) > 12 else "")))
            print("Re-run and choose the other cohort if that is what you meant.")
        return

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    participant_info = load_participant_info()

    if cfg.AVERAGING != "median":
        print("AVERAGING = '%s', not the project default 'median'." % cfg.AVERAGING)
        print("Outputs are tagged '_mean' and are for comparison only.")
        print()

    header = ("%-10s %8s %8s %6s %5s %5s %8s %7s %8s %8s %9s"
              % ("subject", "standups", "unpaired", "short", "ON", "OFF",
                 "outward", "return", "epochs", "wait_min", "wait_med"))
    print(header)
    print("-" * len(header))

    summaries, all_rows, all_problems = [], [], []
    tfr_by_condition = {"ON": [], "OFF": []}
    timecourse_by_condition = {"ON": [], "OFF": []}
    times = None

    for participant, path in participants:
        try:
            summary, rows, problems = inspect_participant(participant, path)
        except Exception as error:
            # Backstop: no single participant may end the run.
            all_problems.append((participant, "inspection failed (%s: %s)"
                                 % (type(error).__name__, error)))
            print("%-10s %s" % (participant, "SKIPPED — inspection failed"))
            continue

        all_rows.extend(rows)
        all_problems.extend((participant, p) for p in problems)

        if summary is None:
            print("%-10s %s" % (participant, "SKIPPED — could not be loaded"))
            continue

        summaries.append(summary)
        print("%-10s %8d %8d %6d %5d %5d %8d %7d %8d %8s %9s"
              % (summary["participant"], summary["standups"], summary["unpaired"],
                 summary["short"], summary["ON"], summary["OFF"],
                 summary["outward"], summary["return"], summary["epochs"],
                 summary["wait_min"], summary["wait_median"]))

        # The participant-id matching in h2_erd can carry its own hx_ assumption.
        # An inspection run should report that and carry on, not abort partway
        # through and lose the table above.
        try:
            roi_channels = roi_for_participant(participant_info, participant)
        except Exception as error:
            all_problems.append(
                (participant,
                 "roi_for_participant() failed (%s: %s) — is this id in %s?"
                 % (type(error).__name__, error,
                    os.path.basename(cfg.PARTICIPANT_INFO))))
            continue

        if roi_channels is None or len(roi_channels) == 0:
            all_problems.append(
                (participant,
                 "no ROI channels returned — is this id in %s?"
                 % os.path.basename(cfg.PARTICIPANT_INFO)))
            continue

        try:
            raw = mne.io.read_raw_fif(path, preload=False)
            erd = participant_erd(raw, find_trials(raw), roi_channels)
        except Exception as error:
            all_problems.append((participant, "ERD preview failed (%s: %s)"
                                 % (type(error).__name__, error)))
            continue

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

    # Reported against LOW_TRIAL_WARNING, not MIN_TRIALS: for the px_ cohort
    # MIN_TRIALS is 0 (those participants are never excluded), so testing
    # against it would silently stop reporting thin sessions in the one cohort
    # where they are most likely.
    below = [s["participant"] for s in summaries
             if min(s["ON"], s["OFF"]) < cfg.LOW_TRIAL_WARNING]
    if below:
        print()
        print("Below %d trials in a condition: %s"
              % (cfg.LOW_TRIAL_WARNING, ", ".join(below)))
        if cfg.MIN_TRIALS == 0:
            print("(%sXX participants are NOT excluded for this — reported only.)"
                  % cfg.SUBJECT_PREFIX)
        else:
            print("(h2_extract.py will exclude these: MIN_TRIALS = %d.)"
                  % cfg.MIN_TRIALS)

    # The cohort tag keeps a px_ inspection from overwriting the hx_ one.
    tag = cfg.COHORT_TAG + output_tag()

    out = os.path.join(cfg.OUTPUT_DIR, "h2_inspection_trials%s.csv" % tag)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print()
    print("Trial-by-trial detail: %s" % out)

    if tfr_by_condition["ON"] and tfr_by_condition["OFF"]:
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
