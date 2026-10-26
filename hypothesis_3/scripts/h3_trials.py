# -*- coding: utf-8 -*-
"""
Trial definition for the Hypothesis 3 pipeline.

Used by both h3_inspect.py and h3_extract.py, so there is only one place where
an ERS trial is defined.
"""

import re

import numpy as np
import mne

import h3_config as cfg


def _code(description):
    """Last run of digits in an annotation description.

    'Stimulus/s2214' -> '2214',  'S1224' -> '1224'
    """
    digits = re.findall(r"\d+", str(description))
    return digits[-1] if digits else ""


def _code_maps():
    """code -> condition, once for each of the three phases."""
    standing = {c: cond for cond, codes in cfg.STANDING_CODES.items() for c in codes}
    onset = {c: cond for cond, codes in cfg.ONSET_CODES.items() for c in codes}
    stop = {c: cond for cond, codes in cfg.STOP_CODES.items() for c in codes}
    return standing, onset, stop


def _direction(code):
    return "outward" if code[2] == "1" else "return"


def find_trials(raw):
    """Match every standing marker to the walk that follows it.

    Returns a list of dicts in time order, one per standing marker found:

        condition      "ON" / "OFF"
        direction      "outward" / "return"
        standing_s     standing marker, seconds from the start of the file
        onset_s        movement-onset marker, seconds (None if unmatched)
        stop_s         movement-stop marker, seconds (None if unmatched)
        wait_s         onset_s - standing_s (None if unmatched)
        walk_s         stop_s - onset_s (None if unmatched)
        onset_sample   sample index of the onset marker, for the baseline epoch
        stop_sample    sample index of the stop marker, for the analysis epoch
        status         "ok" | "unmatched" | "short_wait"

    A standing marker is matched only when the next two target markers are the
    movement onset and then the movement stop of the same condition and the same
    direction. Annotations that are not one of the twelve target codes are
    ignored entirely.
    """
    standing_map, onset_map, stop_map = _code_maps()
    targets = set(standing_map) | set(onset_map) | set(stop_map)

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_code = {code_id: _code(desc) for desc, code_id in event_id.items()}

    markers = [(int(sample), id_to_code[code_id])
               for sample, _, code_id in events
               if id_to_code[code_id] in targets]

    sfreq = raw.info["sfreq"]
    first = raw.first_samp

    trials = []
    i = 0
    while i < len(markers):
        sample, code = markers[i]

        if code not in standing_map:      # a stray onset or stop marker; skip it
            i += 1
            continue

        trial = {
            "condition": standing_map[code],
            "direction": _direction(code),
            "standing_s": (sample - first) / sfreq,
            "onset_s": None,
            "stop_s": None,
            "wait_s": None,
            "walk_s": None,
            "onset_sample": None,
            "stop_sample": None,
            "status": "unmatched",
        }

        if i + 2 < len(markers):
            onset_sample, onset_code = markers[i + 1]
            stop_sample, stop_code = markers[i + 2]

            correct_walk = (
                onset_code in onset_map
                and stop_code in stop_map
                and onset_map[onset_code] == trial["condition"]
                and stop_map[stop_code] == trial["condition"]
                and _direction(onset_code) == trial["direction"]
                and _direction(stop_code) == trial["direction"]
            )
            if correct_walk:
                wait = (onset_sample - sample) / sfreq
                trial["onset_s"] = (onset_sample - first) / sfreq
                trial["stop_s"] = (stop_sample - first) / sfreq
                trial["wait_s"] = wait
                trial["walk_s"] = (stop_sample - onset_sample) / sfreq
                trial["onset_sample"] = onset_sample
                trial["stop_sample"] = stop_sample
                trial["status"] = "ok" if wait >= cfg.MIN_WAIT_S else "short_wait"

        trials.append(trial)
        i += 3 if trial["status"] != "unmatched" else 1

    return trials


def make_events(trials):
    """Two MNE events arrays from the accepted trials, in the same trial order:

        baseline_events  time-locked to movement onset
        task_events      time-locked to movement stop
    """
    accepted = [t for t in trials if t["status"] == "ok"]
    baseline = [[t["onset_sample"], 0, cfg.CONDITION_IDS[t["condition"]]]
                for t in accepted]
    task = [[t["stop_sample"], 0, cfg.CONDITION_IDS[t["condition"]]]
            for t in accepted]
    return (np.array(baseline, dtype=int).reshape(-1, 3),
            np.array(task, dtype=int).reshape(-1, 3))


def count_codes(raw):
    """How many times each of the twelve target codes appears. Used by the
    inspection script to confirm the marker scheme is what we think it is."""
    standing_map, onset_map, stop_map = _code_maps()
    targets = set(standing_map) | set(onset_map) | set(stop_map)

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_code = {code_id: _code(desc) for desc, code_id in event_id.items()}

    counts = {code: 0 for code in sorted(targets)}
    for _, _, code_id in events:
        code = id_to_code[code_id]
        if code in counts:
            counts[code] += 1
    return counts
