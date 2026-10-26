# -*- coding: utf-8 -*-
"""
Inspection for the Hypothesis 4 pipeline.

Reads annotations only. No EEG data is loaded, no spectra are computed and no
artifact rejection is run -- this step exists purely to check how gait epochs
are cut and how long they are.

A gait epoch is the interval from the walking-onset marker to the walking-end
marker of the same condition and direction. The standing period plays no part
in H4 and is not looked at here.

    python h4_inspect.py

Writes into OUTPUT_DIR:
    h4_inspection_trials.csv     every gait epoch and what happened to it
"""

import os

import numpy as np
import pandas as pd
import mne

import h4_config as cfg
from h4_trials import find_trials, count_codes
from h4_spectra import find_participants

mne.set_log_level("ERROR")

# The segment length being considered for AutoReject and for Welch. Only used
# by this report, to show how many trials are too short to contain one.
SEGMENT_S = 2.0


def describe(values):
    """min / median / max of a list, as a compact string."""
    if not values:
        return "%8s %8s %8s" % ("-", "-", "-")
    return "%8.2f %8.2f %8.2f" % (min(values), float(np.median(values)), max(values))


def inspect_participant(participant, path):
    """Gait epochs for one participant, from annotations alone."""
    raw = mne.io.read_raw_fif(path, preload=False)
    trials = find_trials(raw)
    accepted = [t for t in trials if t["status"] == "ok"]

    problems = []
    durations = [t["duration_s"] for t in accepted]
    if durations and min(durations) <= 0:
        problems.append("a gait epoch has zero or negative duration")

    rows = []
    for trial in trials:
        rows.append({
            "participant": participant,
            "condition": trial["condition"],
            "direction": trial["direction"],
            "onset_s": round(trial["onset_s"], 3),
            "end_s": None if trial["end_s"] is None else round(trial["end_s"], 3),
            "duration_s": (None if trial["duration_s"] is None
                           else round(trial["duration_s"], 3)),
            "status": trial["status"],
        })

    # How the duration bounds and a 2 s segment would fall for this participant.
    n = len(durations)
    short = sum(d < cfg.MIN_DURATION_S for d in durations)
    long = sum(d > cfg.MAX_DURATION_S for d in durations)
    under_segment = sum(d < SEGMENT_S for d in durations)

    summary = {
        "participant": participant,
        "onsets": len(trials),
        "unmatched": sum(t["status"] == "unmatched" for t in trials),
        "ON": sum(t["condition"] == "ON" for t in accepted),
        "OFF": sum(t["condition"] == "OFF" for t in accepted),
        "durations": durations,
        "n_short": short,
        "n_long": long,
        "pct_interrupted": (100.0 * (short + long) / n) if n else None,
        "n_under_segment": under_segment,
        "pct_under_segment": (100.0 * under_segment / n) if n else None,
    }
    return summary, rows, problems, count_codes(raw)


def main():
    participants = find_participants()
    if not participants:
        print("No task files found under %s" % cfg.DERIVATIVES_DIR)
        return

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    header = ("%-10s %7s %10s %5s %5s | %-26s | %9s %9s"
              % ("subject", "onsets", "unmatched", "ON", "OFF",
                 "   walk duration (s)", "< %.2fs" % SEGMENT_S, "interrupt"))
    print(header)
    print("%-10s %7s %10s %5s %5s | %8s %8s %8s | %9s %9s"
          % ("", "", "", "", "", "min", "median", "max", "%", "%"))
    print("-" * len(header))

    all_rows, all_problems, summaries = [], [], []
    code_totals = {}
    all_durations = []

    for participant, path in participants:
        summary, rows, problems, codes = inspect_participant(participant, path)
        summaries.append(summary)
        all_rows.extend(rows)
        all_problems.extend((participant, p) for p in problems)
        all_durations.extend(summary["durations"])

        for code, count in codes.items():
            code_totals[code] = code_totals.get(code, 0) + count

        print("%-10s %7d %10d %5d %5d | %s | %9s %9s"
              % (summary["participant"], summary["onsets"], summary["unmatched"],
                 summary["ON"], summary["OFF"], describe(summary["durations"]),
                 "-" if summary["pct_under_segment"] is None
                 else "%.1f" % summary["pct_under_segment"],
                 "-" if summary["pct_interrupted"] is None
                 else "%.1f" % summary["pct_interrupted"]))

    print()
    print("Target markers found across all files")
    for code in sorted(code_totals):
        print("  %s  %6d" % (code, code_totals[code]))

    if all_problems:
        print()
        print("Problems")
        for participant, problem in all_problems:
            print("  %-10s %s" % (participant, problem))

    below = [s["participant"] for s in summaries
             if min(s["ON"], s["OFF"]) < cfg.MIN_TRIALS]
    if below:
        print()
        print("Below %d gait epochs in a condition: %s"
              % (cfg.MIN_TRIALS, ", ".join(below)))

    if all_durations:
        durations = np.array(all_durations, dtype=float)
        total = len(durations)
        print()
        print("Walk duration across all %d gait epochs" % total)
        print("  min      %.2f s" % durations.min())
        print("  median   %.2f s" % np.median(durations))
        print("  max      %.2f s" % durations.max())
        print("  10th pct %.2f s   90th pct %.2f s"
              % (np.percentile(durations, 10), np.percentile(durations, 90)))

        short = int(np.sum(durations < cfg.MIN_DURATION_S))
        long = int(np.sum(durations > cfg.MAX_DURATION_S))
        print()
        print("Interrupted trials, bounds [%.2f, %.2f] s"
              % (cfg.MIN_DURATION_S, cfg.MAX_DURATION_S))
        print("  shorter than %.2f s: %d (%.1f%%)"
              % (cfg.MIN_DURATION_S, short, 100.0 * short / total))
        print("  longer than %.2f s:  %d (%.1f%%)"
              % (cfg.MAX_DURATION_S, long, 100.0 * long / total))
        print("  retained:            %d (%.1f%%)"
              % (total - short - long, 100.0 * (total - short - long) / total))

        # What a SEGMENT_S segment would give, over the trials that survive the
        # bounds -- those are the only ones the main script will segment.
        kept = durations[(durations >= cfg.MIN_DURATION_S)
                         & (durations <= cfg.MAX_DURATION_S)]
        print()
        print("If retained trials are segmented at %.2f s" % SEGMENT_S)
        under = int(np.sum(kept < SEGMENT_S))
        print("  too short for one segment: %d (%.1f%%)"
              % (under, 100.0 * under / len(kept) if len(kept) else 0.0))
        segments = np.floor(kept / SEGMENT_S).astype(int)
        for count in range(0, int(segments.max()) + 1 if len(segments) else 1):
            share = 100.0 * np.mean(segments == count)
            if share > 0:
                print("  %d segment(s): %5.1f%% of retained trials" % (count, share))
        print("  4 or more segments (where a fraction rule would differ from")
        print("  any-flag): %.1f%%" % (100.0 * np.mean(segments >= 4)))

    out = os.path.join(cfg.OUTPUT_DIR, "h4_inspection_trials.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print()
    print("Trial-by-trial detail: %s" % out)


if __name__ == "__main__":
    main()
