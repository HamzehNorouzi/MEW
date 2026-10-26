# -*- coding: utf-8 -*-
"""
Gait epoch definition for the Hypothesis 4 pipeline.

Used by both h4_extract.py and h4_inspect.py, so there is only one place where
a gait epoch is defined.
"""

import re

import numpy as np
import mne

import h4_config as cfg


def _code(description):
    """Last run of digits in an annotation description.

    'Stimulus/s2213' -> '2213',  'S1224' -> '1224'
    """
    digits = re.findall(r"\d+", str(description))
    return digits[-1] if digits else ""


def _code_maps():
    """code -> condition, once for onset codes and once for end codes."""
    onset = {c: cond for cond, codes in cfg.ONSET_CODES.items() for c in codes}
    end = {c: cond for cond, codes in cfg.END_CODES.items() for c in codes}
    return onset, end


def _direction(code):
    return "outward" if code[2] == "1" else "return"


def find_trials(raw):
    """Pair every gait-onset marker with the gait-end marker that follows it.

    Returns a list of dicts in time order, one per onset marker found:

        condition      "ON" / "OFF"
        direction      "outward" / "return"
        onset_s        gait onset, seconds from the start of the file
        end_s          gait end, seconds (None if unmatched)
        duration_s     end_s - onset_s (None if unmatched)
        onset_sample   sample index of the onset marker
        end_sample     sample index of the end marker
        status         "ok" | "unmatched"

    An onset is matched only when the next target marker is the gait end of the
    same condition and the same direction. Annotations that are not one of the
    eight target codes are ignored entirely.
    """
    onset_map, end_map = _code_maps()
    targets = set(onset_map) | set(end_map)

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

        if code not in onset_map:          # a stray end marker; skip it
            i += 1
            continue

        trial = {
            "condition": onset_map[code],
            "direction": _direction(code),
            "onset_s": (sample - first) / sfreq,
            "end_s": None,
            "duration_s": None,
            "onset_sample": sample,
            "end_sample": None,
            "status": "unmatched",
        }

        if i + 1 < len(markers):
            end_sample, end_code = markers[i + 1]
            correct_end = (end_code in end_map
                           and end_map[end_code] == trial["condition"]
                           and _direction(end_code) == trial["direction"])
            if correct_end:
                trial["end_s"] = (end_sample - first) / sfreq
                trial["duration_s"] = (end_sample - sample) / sfreq
                trial["end_sample"] = end_sample
                trial["status"] = "ok"

        trials.append(trial)
        i += 2 if trial["status"] == "ok" else 1

    return trials


def make_events(trials):
    """MNE events array from the accepted trials, time-locked to gait onset."""
    rows = [[t["onset_sample"], 0, cfg.CONDITION_IDS[t["condition"]]]
            for t in trials if t["status"] == "ok"]
    return np.array(rows, dtype=int).reshape(-1, 3)


def count_codes(raw):
    """How many times each of the eight target codes appears, so a wrong or
    missing trigger code is visible."""
    onset_map, end_map = _code_maps()
    targets = set(onset_map) | set(end_map)

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_code = {code_id: _code(desc) for desc, code_id in event_id.items()}

    counts = {code: 0 for code in sorted(targets)}
    for _, _, code_id in events:
        code = id_to_code[code_id]
        if code in counts:
            counts[code] += 1
    return counts
