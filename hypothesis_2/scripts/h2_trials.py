# -*- coding: utf-8 -*-
"""
Trial definition for the Hypothesis 2 pipeline.

Both the inspection script and the extraction script call find_trials(), so
there is only one place where a trial is defined.
"""

import re

import numpy as np
import mne

import h2_config as cfg


def _code(description):
    """Last run of digits in an annotation description.

    'Stimulus/s2212' -> '2212',  'S1213' -> '1213'
    """
    digits = re.findall(r"\d+", str(description))
    return digits[-1] if digits else ""


def _code_maps():
    """code -> condition, once for stand-up codes and once for onset codes."""
    standup = {c: cond for cond, codes in cfg.STANDUP_CODES.items() for c in codes}
    onset = {c: cond for cond, codes in cfg.ONSET_CODES.items() for c in codes}
    return standup, onset


def _direction(code):
    return "outward" if code[2] == "1" else "return"


def find_trials(raw):
    """Pair every stand-up marker with the walk that follows it.

    Returns a list of dicts in time order, one per stand-up marker found:

        condition      "ON" / "OFF"
        direction      "outward" / "return"
        standup_s      stand-up marker, seconds from the start of the file
        onset_s        movement-onset marker, seconds (None if unpaired)
        wait_s         onset_s - standup_s (None if unpaired)
        onset_sample   sample index of the onset marker, for building events
        status         "ok" | "unpaired" | "short_wait"

    A stand-up is paired only when the very next target marker is the movement
    onset of the same condition and the same direction. Annotations that are not
    one of the eight target codes are ignored entirely.
    """
    standup_map, onset_map = _code_maps()
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_code = {code_id: _code(desc) for desc, code_id in event_id.items()}

    markers = [(int(sample), id_to_code[code_id])
               for sample, _, code_id in events
               if id_to_code[code_id] in standup_map or id_to_code[code_id] in onset_map]

    sfreq = raw.info["sfreq"]
    first = raw.first_samp

    trials = []
    i = 0
    while i < len(markers):
        sample, code = markers[i]

        if code not in standup_map:        # a stray onset marker; skip it
            i += 1
            continue

        trial = {
            "condition": standup_map[code],
            "direction": _direction(code),
            "standup_s": (sample - first) / sfreq,
            "onset_s": None,
            "wait_s": None,
            "onset_sample": None,
            "status": "unpaired",
        }

        if i + 1 < len(markers):
            next_sample, next_code = markers[i + 1]
            correct_walk = (next_code in onset_map
                            and onset_map[next_code] == trial["condition"]
                            and _direction(next_code) == trial["direction"])
            if correct_walk:
                wait = (next_sample - sample) / sfreq
                trial["onset_s"] = (next_sample - first) / sfreq
                trial["wait_s"] = wait
                trial["onset_sample"] = next_sample
                trial["status"] = "ok" if wait >= cfg.MIN_WAIT_S else "short_wait"

        trials.append(trial)
        i += 2 if trial["status"] != "unpaired" else 1

    return trials


def make_events(trials):
    """MNE events array from the accepted trials, time-locked to movement onset."""
    rows = [[t["onset_sample"], 0, cfg.CONDITION_IDS[t["condition"]]]
            for t in trials if t["status"] == "ok"]
    return np.array(rows, dtype=int).reshape(-1, 3)
