#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STAGE 0 — SHARED EEG PREPROCESSING
BeechBand vibrotactile stimulation study
================================================================================

Applies the shared preprocessing chain to every recording file, independently.
Output feeds Pipelines A (resting-state / H1), B (ERD+ERS / H2,H3) and
C (gait beta / H4), which are NOT implemented here.

Chain, per recording file
-------------------------
  load  ->  keep+retype EOG, drop mastoids  ->  montage (.elc, auto-oriented)
        ->  annotation inventory  ->  resample 250 Hz  ->  detrend
        ->  band-pass 0.1-100 Hz (no notch)  ->  average reference  ->  ASR
        ->  ICA (extended Infomax)  ->  ICLabel + EOG-correlation
        ->  reject union of both  ->  QC gate  ->  save

The full continuous recording is processed. NO cropping or block separation
happens here; annotations are inventoried for the log and carried through to
the cleaned output so Pipeline A can extract blocks later.

Resume
------
Manifest-driven at the granularity of one recording file. Re-running the script
skips anything already marked `done`. Use --force to reprocess.

Usage
-----
  # 1. Look at the data first — no processing, answers open questions
  python stage0_preprocess.py --root /path/to/main_folder --inventory

  # 2. Full run
  python stage0_preprocess.py --root /path/to/main_folder

  # 3. Resume after an interruption (identical command; done rows are skipped)
  python stage0_preprocess.py --root /path/to/main_folder

  # 4. Reprocess specific subjects
  python stage0_preprocess.py --root /path/to/main_folder --subjects hx_04 hx_07 --force

Requirements
------------
  pip install "mne>=1.6" mne-icalabel meegkit numpy scipy pandas
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# MNE is imported lazily inside main() so that --help works without it.

# ==============================================================================
# CONFIG — everything you may need to change lives in this block
# ==============================================================================
asr_arg = input("Enable ASR? True/False: ")
loaded_sample = input("loading px or hx?")

if loaded_sample == "hx":
    reject_ic_threshold = 0.8
else:
    reject_ic_threshold = 0.96

CONFIG: dict[str, Any] = {

    # ---- Subjects -----------------------------------------------------------
    # Skipped entirely, logged as `excluded_by_request`.
    "EXCLUDE_SUBJECTS": [],
    "SUBJECT_PATTERN": r"^px_\d+$",

    # ---- Layout -------------------------------------------------------------
    # main_folder/hx_XX/rest/  and  main_folder/hx_XX/task/
    # Recording stems are matched case-insensitively as substrings.
    "RECORDINGS": {
        "pre_task_rest":  {"subdir": "rest", "stem": "pre_task_rest",  "kind": "rest"},
        "post_task_rest": {"subdir": "rest", "stem": "post_task_rest", "kind": "rest"},
        "task": {"subdir": "task", "stem": "task", "kind": "task"},
    },
    # Tried in order. First extension with any match wins.
    "RAW_EXTENSIONS": [".vhdr"],

    # ---- Montage ------------------------------------------------------------
    # .elc file located in main_folder. Set to an explicit filename to override.
    "ELC_FILENAME": "standard_waveguard64.elc",          # e.g. "waveguard64.elc"; None = auto-discover
    "ELC_UNITS": "auto",           # "auto" | "mm" | "m"
    "MONTAGE_MATCH_CASE": False,   # case-insensitive channel-name matching

    # ---- Montage orientation ------------------------------------------------
    # MNE expects RAS head coordinates: +X right, +Y anterior, +Z superior.
    # ANT/ASA .elc files are frequently written ALS (+X anterior, +Y left,
    # +Z up). Read as RAS that puts frontal electrodes on the right and
    # posterior ones on the left — a 90 degree rotation.
    #   "auto"       infer the frame from anatomical landmarks and correct it
    #                (recommended: self-verifying, and cannot mirror L/R)
    #   "none"       trust the file as written
    #   "als_to_ras" apply the fixed ALS -> RAS transform without checking
    "MONTAGE_ORIENTATION": "none",
    # Landmarks used by "auto". Only those present are used.
    "MONTAGE_LANDMARKS": {
        "anterior":  ["Fpz", "Fp1", "Fp2", "AFz", "AF3", "AF4", "AF7", "AF8", "Fz"],
        "posterior": ["Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8", "Pz"],
        "left":      ["T7", "T3", "C5", "FT7", "TP7", "P7", "T5", "F7"],
        "right":     ["T8", "T4", "C6", "FT8", "TP8", "P8", "T6", "F8"],
    },

    # ---- Channels to drop ---------------------------------------------------
    # Matched case-insensitively against channel names. Anything not present is
    # logged as a warning rather than raising, so the run does not die on a
    # naming mismatch.
    # EOG is KEPT and retyped as an 'eog' channel. It is then used to identify
    # ocular independent components by correlating each IC's time course with
    # it. Being typed 'eog' it is automatically excluded from the average
    # reference, from ASR, and from the ICA decomposition itself — MNE picks
    # data channels only for all three.
    "EOG_CHANNELS": ["EOG"],
    # These ARE dropped.
    "MASTOID_CHANNELS": ["M1", "M2"],
    "EXPECTED_N_EEG_AFTER_DROP": 61,
    "EXPECTED_N_EOG": 1,
    # Keep the EOG channel in the saved output. Harmless downstream: it is typed
    # 'eog', so anything that picks EEG will skip it. Set True to drop it after
    # ICA has been applied.
    "DROP_EOG_AFTER_ICA": False,

    # ---- Edge cropping ------------------------------------------------------
    # Everything before the first marker and after the last marker is
    # participant preparation / pack-down, not recording. Dropped with a margin.
    "CROP_TO_MARKERS": True,
    "CROP_EDGE_MARGIN_S": 1.0,
    # Which annotations define "first" and "last". For rest files only the
    # canonical block codes count, so a stray preparation trigger cannot defeat
    # the crop. For task files every annotation counts (trigger scheme TBC).
    "CROP_EDGE_CODES": {"rest": ["1002", "1004"], "task": None},   # None = any

    # ---- Break removal (task files only) ------------------------------------
    # During a trial there are several phase markers, and a trial lasts ~30 s,
    # so within- and between-trial marker gaps are only a few seconds. A silence
    # of 30 s or more is therefore a break, not part of the protocol.
    #
    # Breaks are only ever taken after movement has terminated and the ERS
    # window has closed, and the gap closes on the first phase marker of the
    # next trial — which precedes that trial's baseline window. No analysis
    # window can fall inside the gap, so the full gap is removed and the
    # margins below default to zero. Raise them only if you want to hold the
    # concatenation seam further away from analysis data.
    "BREAK_REMOVAL_ENABLED": True,
    "BREAK_KINDS": ["task"],        # never applied to rest files
    "BREAK_MIN_GAP_S": 30.0,
    "BREAK_KEEP_AFTER_S": 0.0,      # kept after the marker that opens the gap
    "BREAK_KEEP_BEFORE_S": 0.0,     # kept before the marker that closes it
    "BREAK_MIN_DROP_S": 5.0,        # skip if the interior is shorter than this
    # Subtract each surviving segment's own mean before concatenation. Removes
    # the DC step at each join, so the seam is small and the 0.1 Hz FIR — which
    # is ~33 s long at 250 Hz — has far less to ring across. Harmless: the
    # band-pass and average reference remove DC regardless.
    "BREAK_MATCH_SEGMENT_DC": True,

    # ---- Filtering ----------------------------------------------------------
    "RESAMPLE_HZ": 250.0,
    "L_FREQ": 0.1,
    "H_FREQ": 100.0,
    "FIR_DESIGN": "firwin",
    "FILTER_PHASE": "zero",
    "APPLY_NOTCH": False,          # per instruction — no notch filter
    "NOTCH_FREQS": [50.0, 100.0],  # only used if APPLY_NOTCH is True
    "DETREND_TYPE": "linear",

    # ---- ASR ----------------------------------------------------------------
    "ASR_ENABLED": asr_arg.lower() == "true",
    "ASR_CUTOFF": 20,
    "ASR_METHOD": "euclid",        # "euclid" | "riemann"
    # ASR's calibration normally assumes a ~1-2 Hz high-pass. Set this to e.g.
    # 1.0 to calibrate on a high-passed copy while correcting the 0.1 Hz data.
    # None = calibrate and correct on the 0.1 Hz data (faithful to the methods).
    "ASR_CALIB_HPF": None,

    # ---- ICA ----------------------------------------------------------------
    "ICA_N_COMPONENTS": 60,        # capped at the estimated rank
    "ICA_METHOD": "infomax",
    "ICA_EXTENDED": True,          # required by ICLabel
    "ICA_RANDOM_STATE": 97,
    "ICA_MAX_ITER": "auto",
    # ICLabel is trained on 1-100 Hz data. Fitting ICA on a 1 Hz high-passed
    # copy and applying the solution to the 0.1 Hz data is standard practice and
    # does not change the analysed frequency band. Set to None to fit on the
    # 0.1 Hz data directly (ICLabel will then warn and may be less reliable).
    "ICA_FIT_HPF": 1.0,

    # ---- ICLabel ------------------------------------------------------------
    # Methods say "non-brain components will be rejected", so only `brain` is
    # kept. The log also records what ("brain", "other") would have given, since
    # keeping the ambiguous `other` class is the more common convention.
    "KEEP_LABELS": ["brain"], # keep "other" as rank of file might be lower than 60, therefore makes more ICs
    "ICLABEL_PROB_THRESHOLD": 0.0,  # 0.0 = accept ICLabel's argmax as-is

    # ---- EOG-based component detection --------------------------------------
    # Correlates every IC time course with the EOG channel and flags outliers.
    # Runs IN ADDITION to ICLabel; the rejected set is the union of the two, so
    # this can only ever remove more, never fewer, components.
    "EOG_DETECT_ENABLED": True,
    "EOG_DETECT_MEASURE": "zscore",     # "zscore" | "correlation"
    "EOG_DETECT_THRESHOLD": 3.0,        # z > 3.0, or |r| > threshold if "correlation"

    # ---- QC gate ------------------------------------------------------------
    "MAX_IC_REJECT_FRACTION": reject_ic_threshold,   # >= 80% rejected -> excluded_noisy

    # ---- Rest-state annotation inventory (log only) -------------------------
    "REST_CODES": {"1002": "eyes_closed_stim_off", "1004": "eyes_closed_stim_on"},
    "MIN_BLOCK_DURATION_S": 59.0,
    "EXPECTED_ANNOTATIONS_PER_REST_FILE": 8,
    "EXPECTED_BLOCKS_PER_CODE": 2,

    # ---- Output -------------------------------------------------------------
    "DERIVATIVES_DIRNAME": "derivatives",
    "SAVE_ICA": True,
    "FIF_OVERWRITE": True,
}

MANIFEST_COLUMNS = [
    "subject", "recording", "kind", "status",
    "n_input_files", "input_files",
    "duration_s", "sfreq_in", "sfreq_out",
    "n_chan_in", "n_eeg_kept", "n_eog_kept", "eog_channels", "dropped_channels",
    "duration_raw_s", "crop_start_s", "crop_end_s", "duration_after_edge_crop_s",
    "n_breaks_dropped", "break_total_s", "duration_after_breaks_s",
    "gap_p50_s", "gap_p95_s", "gap_max_s",
    "rank_est", "n_ica", "n_ic_rejected", "pct_ic_rejected",
    "n_ic_iclabel", "n_ic_eog", "n_ic_eog_only",
    "asr_pct_reconstructed", "eog_eeg_corr_pre_asr", "eog_eeg_corr_post_asr",
    "n_ann_1002", "n_ann_1004", "n_blocks_valid", "n_blocks_short",
    "flags", "error", "timestamp_utc",
]

log = logging.getLogger("stage0")


# ==============================================================================
# Small helpers
# ==============================================================================

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(logfile: Path, verbose: bool = True) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    # Some dependencies attach a root handler on import; without this every
    # line would be printed twice.
    log.propagate = False

    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
    log.addHandler(sh)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract_code(description: str) -> str | None:
    """Pull a numeric trigger code out of an annotation description.

    Handles bare '1002', 'Stimulus/S1002', 'Trigger 1002', 'S1002', etc.
    """
    m = re.findall(r"\d+", str(description))
    return m[-1] if m else None


# ==============================================================================
# Discovery
# ==============================================================================

def find_subjects(root: Path) -> list[str]:
    pat = re.compile(CONFIG["SUBJECT_PATTERN"])
    subs = sorted(p.name for p in root.iterdir() if p.is_dir() and pat.match(p.name))
    return subs


def find_recording_files(root: Path, subject: str, rec_key: str) -> list[Path]:
    """Locate the raw file(s) for one recording of one subject.

    Searches <subject>/<subdir>/ first, then the whole subject folder.
    Returns files sorted by name; more than one means split recordings that
    will be concatenated.
    """
    spec = CONFIG["RECORDINGS"][rec_key]
    sub_dir = root / subject
    search_dirs = [sub_dir / spec["subdir"], sub_dir]

    stem_n = norm(spec["stem"])
    # 'task' would otherwise match 'pre_task_rest'; exclude the rest stems.
    other_stems = [norm(v["stem"]) for k, v in CONFIG["RECORDINGS"].items()
                   if k != rec_key and norm(v["stem"]) != stem_n]

    for ext in CONFIG["RAW_EXTENSIONS"]:
        hits: list[Path] = []
        for d in search_dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob(f"*{ext}")):
                fn = norm(f.stem)
                if stem_n not in fn:
                    continue
                # reject if a longer, more specific stem also matches
                if any(o in fn for o in other_stems if len(o) > len(stem_n)):
                    continue
                hits.append(f)
            if hits:
                break
        if hits:
            # de-duplicate while preserving order
            seen, out = set(), []
            for h in hits:
                if h.resolve() not in seen:
                    seen.add(h.resolve())
                    out.append(h)
            return out
    return []


def find_elc(root: Path) -> Path | None:
    if CONFIG["ELC_FILENAME"]:
        p = root / CONFIG["ELC_FILENAME"]
        return p if p.exists() else None
    hits = sorted(root.glob("*.elc"))
    if not hits:
        hits = sorted(root.rglob("*.elc"))
    return hits[0] if hits else None


# ==============================================================================
# Edge cropping and break removal — run BEFORE any filtering
# ==============================================================================

def is_structural_annotation(desc: str) -> bool:
    """MNE's own bookkeeping annotations, not protocol markers.

    raw.crop() inserts 'EDGE boundary' and concatenate_raws inserts
    'BAD boundary'. Counting these as protocol markers would let a previous
    crop define the edges of the next one.
    """
    d = str(desc).strip().lower()
    return d.startswith("bad") or "boundary" in d


def marker_times(raw, kind: str) -> np.ndarray:
    """Onsets of the annotations that count as protocol markers, expressed on
    the same timescale as raw.times.

    IMPORTANT: annotation onsets stay on the original recording timeline after a
    crop, while raw.crop() and raw.times are relative to the current start.
    Mixing the two silently removes the wrong spans, so first_time is subtracted
    here and every caller works in times-relative seconds.
    """
    codes = CONFIG["CROP_EDGE_CODES"].get(kind)
    want = None if codes is None else set(codes)
    offset = float(raw.first_time)

    t = []
    for o, d in zip(raw.annotations.onset, raw.annotations.description):
        if is_structural_annotation(d):
            continue
        if want is not None and extract_code(str(d)) not in want:
            continue
        t.append(float(o) - offset)
    return np.sort(np.asarray(t, dtype=float))


def gap_stats(times: np.ndarray) -> dict[str, Any]:
    """Distribution of intervals between consecutive markers.

    Reported so the break threshold can be checked against the data rather than
    taken on trust: if breaks are real, they sit far above the p95 of ordinary
    within- and between-trial gaps.
    """
    if times.size < 2:
        return {"n_gaps": 0}
    g = np.diff(times)
    return {
        "n_gaps": int(g.size),
        "p50": round(float(np.percentile(g, 50)), 2),
        "p95": round(float(np.percentile(g, 95)), 2),
        "p99": round(float(np.percentile(g, 99)), 2),
        "max": round(float(g.max()), 2),
        "gaps_over_10s": [round(float(x), 2) for x in np.sort(g[g > 10.0])[::-1][:25]],
    }


def crop_to_markers(raw, kind: str, flags: list[str]) -> dict[str, Any]:
    """Drop everything before the first marker and after the last, with a margin.

    Removes the participant-preparation and pack-down periods, which carry large
    non-task artifacts that would otherwise contaminate ASR calibration and the
    ICA decomposition.
    """
    rep: dict[str, Any] = {"applied": False}
    if not CONFIG["CROP_TO_MARKERS"]:
        return rep

    t = marker_times(raw, kind)
    dur = raw.n_times / raw.info["sfreq"]
    rep["duration_before_s"] = round(dur, 2)

    if t.size == 0:
        msg = "no markers found — edge cropping skipped, full recording kept"
        flags.append(msg)
        log.warning(f"    {msg}")
        rep["reason"] = "no markers"
        return rep

    margin = CONFIG["CROP_EDGE_MARGIN_S"]
    tmin = max(0.0, float(t[0]) - margin)
    tmax = min(raw.times[-1], float(t[-1]) + margin)

    if tmax <= tmin:
        flags.append("edge crop would leave nothing — skipped")
        rep["reason"] = "degenerate window"
        return rep

    raw.crop(tmin=tmin, tmax=tmax)

    rep.update(applied=True, first_marker_s=round(float(t[0]), 2),
               last_marker_s=round(float(t[-1]), 2), margin_s=margin,
               crop_start_s=round(tmin, 2), crop_end_s=round(tmax, 2),
               dropped_head_s=round(tmin, 2), dropped_tail_s=round(dur - tmax, 2),
               duration_after_s=round(raw.n_times / raw.info["sfreq"], 2))
    log.info(f"    edge crop: kept {tmin:.1f}-{tmax:.1f}s "
             f"(dropped {rep['dropped_head_s']:.1f}s head + "
             f"{rep['dropped_tail_s']:.1f}s tail of {dur:.1f}s)")
    return rep


def find_breaks(raw, kind: str, flags: list[str]) -> tuple[list[tuple[float, float]],
                                                           dict[str, Any]]:
    """Locate break periods as long marker-free gaps. Task files only.

    Only the interior of each gap is returned: BREAK_KEEP_AFTER_S is preserved
    after the marker that opens it and BREAK_KEEP_BEFORE_S before the one that
    closes it, so the analysis windows of the adjacent trials survive.
    """
    rep: dict[str, Any] = {"enabled": CONFIG["BREAK_REMOVAL_ENABLED"], "kind": kind}

    if not CONFIG["BREAK_REMOVAL_ENABLED"] or kind not in CONFIG["BREAK_KINDS"]:
        rep["applied"] = False
        rep["reason"] = f"not applied to '{kind}' files"
        return [], rep

    t = marker_times(raw, kind)
    rep["gap_stats"] = gap_stats(t)
    if t.size < 2:
        rep["applied"] = False
        rep["reason"] = "fewer than 2 markers"
        return [], rep

    thr = CONFIG["BREAK_MIN_GAP_S"]
    keep_a, keep_b = CONFIG["BREAK_KEEP_AFTER_S"], CONFIG["BREAK_KEEP_BEFORE_S"]
    min_drop = CONFIG["BREAK_MIN_DROP_S"]

    breaks: list[tuple[float, float]] = []
    detail: list[dict[str, Any]] = []
    for a, b in zip(t[:-1], t[1:]):
        gap = float(b - a)
        if gap < thr:
            continue
        s, e = float(a) + keep_a, float(b) - keep_b
        entry = {"gap_start_s": round(float(a), 2), "gap_end_s": round(float(b), 2),
                 "gap_s": round(gap, 2), "drop_start_s": round(s, 2),
                 "drop_end_s": round(e, 2), "drop_s": round(max(0.0, e - s), 2)}
        if e - s < min_drop:
            entry["dropped"] = False
            entry["reason"] = f"interior {e - s:.1f}s < {min_drop}s after margins"
        else:
            entry["dropped"] = True
            breaks.append((s, e))
        detail.append(entry)

    rep.update(applied=bool(breaks), threshold_s=thr,
               keep_after_s=keep_a, keep_before_s=keep_b,
               n_candidates=len(detail), n_dropped=len(breaks),
               total_dropped_s=round(sum(e - s for s, e in breaks), 2),
               breaks=detail)

    if detail:
        log.info(f"    breaks: {len(breaks)} dropped of {len(detail)} gap(s) "
                 f">= {thr}s, total {rep['total_dropped_s']:.1f}s removed")
        for d in detail:
            mark = "DROP" if d["dropped"] else "keep"
            log.info(f"      {mark}  gap {d['gap_start_s']:8.1f}-{d['gap_end_s']:8.1f}s "
                     f"({d['gap_s']:6.1f}s) -> remove {d['drop_s']:6.1f}s")
    else:
        log.info(f"    breaks: none (no marker gap >= {thr}s)")

    return breaks, rep


def drop_breaks(raw, breaks: list[tuple[float, float]], flags: list[str]):
    """Remove break spans by cropping the recording into the surviving segments
    and concatenating them.

    concatenate_raws inserts a 'BAD boundary' annotation at every join, and MNE's
    epoching honours BAD_* spans by default — so epochs straddling a join are
    dropped downstream automatically. Filter ringing across those joins is
    therefore contained rather than silently analysed.
    """
    import mne

    if not breaks:
        return raw, {"applied": False}

    sf = raw.info["sfreq"]
    dur = raw.n_times / sf

    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(breaks):
        s, e = max(0.0, s), min(dur, e)
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < dur:
        keep.append((cursor, dur))

    keep = [(a, b) for a, b in keep if (b - a) > 1.0 / sf]
    if not keep:
        flags.append("break removal would leave no data — skipped")
        log.error("    break removal would leave no data; skipped")
        return raw, {"applied": False, "reason": "nothing would remain"}

    segs = []
    for a, b in keep:
        seg = raw.copy().crop(tmin=a, tmax=min(b, raw.times[-1]))
        if CONFIG["BREAK_MATCH_SEGMENT_DC"]:
            seg.apply_function(lambda x: x - x.mean(), picks="all",
                               channel_wise=True, verbose="ERROR")
        segs.append(seg)
    out = mne.concatenate_raws(segs, verbose="ERROR") if len(segs) > 1 else segs[0]

    rep = {"applied": True, "n_segments_kept": len(keep),
           "segments_s": [[round(a, 2), round(b, 2)] for a, b in keep],
           "duration_before_s": round(dur, 2),
           "duration_after_s": round(out.n_times / sf, 2)}
    log.info(f"    breaks removed: {len(keep)} segment(s) kept, "
             f"{dur:.1f}s -> {rep['duration_after_s']:.1f}s "
             f"({len(keep) - 1} boundary marker(s) inserted)")
    return out, rep


# ==============================================================================
# Annotation inventory  (log only — no cropping in Stage 0)
# ==============================================================================

@dataclass
class Block:
    code: str
    label: str
    onset: float
    offset: float
    duration: float
    contains_other_code: bool
    valid: bool
    reason: str = ""


def inventory_annotations(raw, kind: str) -> dict[str, Any]:
    """Catalogue every annotation, and for rest files derive candidate blocks.

    A candidate block is the interval between two CONSECUTIVE annotations of the
    same code that contains no annotation of the other rest code. That single
    rule handles both the clean 8-annotation case and files with extra
    annotations from aborted blocks:

      clean   1004 @ [s1, e1, s2, e2]
              (s1,e1) valid | (e1,s2) contains 1002 -> rejected | (s2,e2) valid

      aborted 1004 @ [0, 12, 20, 80]
              (0,12)=12s short | (12,20)=8s short | (20,80)=60s valid
    """
    descs = [str(d) for d in raw.annotations.description]
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durs = np.asarray(raw.annotations.duration, dtype=float)

    codes = [extract_code(d) for d in descs]

    # Full verbatim inventory — this is how the task trigger scheme gets read
    # off the logs before Pipeline B is written.
    all_counts: dict[str, int] = {}
    for d in descs:
        all_counts[d] = all_counts.get(d, 0) + 1

    inv: dict[str, Any] = {
        "n_annotations_total": len(descs),
        "unique_descriptions": dict(sorted(all_counts.items(), key=lambda kv: -kv[1])),
        "first_10": [
            {"description": d, "code": c, "onset_s": round(float(o), 4),
             "duration_s": round(float(du), 4)}
            for d, c, o, du in list(zip(descs, codes, onsets, durs))[:10]
        ],
        "blocks": [],
        "n_blocks_valid": 0,
        "n_blocks_short": 0,
        "flags": [],
    }

    rest_codes = CONFIG["REST_CODES"]
    for code in rest_codes:
        inv[f"n_ann_{code}"] = sum(1 for c in codes if c == code)

    if kind != "rest":
        return inv

    total_rest_ann = sum(inv[f"n_ann_{c}"] for c in rest_codes)
    expected = CONFIG["EXPECTED_ANNOTATIONS_PER_REST_FILE"]
    if total_rest_ann != expected:
        inv["flags"].append(
            f"expected {expected} rest annotations, found {total_rest_ann} "
            f"(aborted blocks likely)"
        )

    min_dur = CONFIG["MIN_BLOCK_DURATION_S"]
    candidates: list[Block] = []

    for code, label in rest_codes.items():
        idx = [i for i, c in enumerate(codes) if c == code]
        idx.sort(key=lambda i: onsets[i])
        other_onsets = np.array(
            [onsets[i] for i, c in enumerate(codes) if c in rest_codes and c != code],
            dtype=float,
        )

        for a, b in zip(idx[:-1], idx[1:]):
            t0, t1 = float(onsets[a]), float(onsets[b])
            dur = t1 - t0
            contains_other = bool(np.any((other_onsets > t0) & (other_onsets < t1)))

            if contains_other:
                reason, valid = "interval spans a block of the other condition", False
            elif dur < min_dur:
                reason, valid = f"duration {dur:.1f}s < {min_dur}s", False
            else:
                reason, valid = "", True

            candidates.append(Block(code, label, t0, t1, dur, contains_other, valid, reason))

    # Greedy earliest-first selection among valid, non-overlapping candidates.
    selected: list[Block] = []
    for b in sorted([c for c in candidates if c.valid], key=lambda c: c.onset):
        if all(b.onset >= s.offset or b.offset <= s.onset for s in selected):
            selected.append(b)
        else:
            b.valid = False
            b.reason = "overlaps an earlier selected block"

    sel_ids = {(b.code, b.onset, b.offset) for b in selected}
    for b in candidates:
        if b.valid and (b.code, b.onset, b.offset) not in sel_ids:
            b.valid = False
            b.reason = b.reason or "not selected"

    inv["blocks"] = [
        {**asdict(b), "onset": round(b.onset, 4), "offset": round(b.offset, 4),
         "duration": round(b.duration, 4)}
        for b in sorted(candidates, key=lambda c: c.onset)
    ]
    inv["n_blocks_valid"] = len(selected)
    inv["n_blocks_short"] = sum(
        1 for b in candidates if not b.valid and not b.contains_other_code
    )

    per_code: dict[str, int] = {}
    for b in selected:
        per_code[b.code] = per_code.get(b.code, 0) + 1
    inv["valid_blocks_per_code"] = per_code

    exp_blocks = CONFIG["EXPECTED_BLOCKS_PER_CODE"]
    for code, label in rest_codes.items():
        n = per_code.get(code, 0)
        if n != exp_blocks:
            inv["flags"].append(
                f"{label} (code {code}): {n} valid block(s), expected {exp_blocks}"
            )
        if n == 0:
            inv["flags"].append(f"NO usable data for {label} — rest data unusable")

    return inv


# ==============================================================================
# Preprocessing steps
# ==============================================================================

def load_raw(paths: list[Path]):
    """Read one or more raw files and concatenate. Annotation onsets are
    shifted automatically by MNE's concatenate_raws."""
    import mne

    raws = []
    for p in paths:
        ext = p.suffix.lower()
        log.info(f"    reading {p.name}")
        if ext == ".cnt":
            # ANT Neuro. read_raw_ant is the modern reader (MNE >= 1.9);
            # read_raw_cnt is the Neuroscan-style fallback.
            try:
                r = mne.io.read_raw_ant(p, preload=True, verbose="ERROR")
            except (AttributeError, Exception) as e:  # noqa: BLE001
                log.debug(f"      read_raw_ant unavailable/failed ({e}); trying read_raw_cnt")
                r = mne.io.read_raw_cnt(p, preload=True, verbose="ERROR")
        elif ext == ".vhdr":
            r = mne.io.read_raw_brainvision(p, preload=True, verbose="ERROR")
        elif ext == ".set":
            r = mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR")
        elif ext == ".edf":
            r = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        elif ext == ".bdf":
            r = mne.io.read_raw_bdf(p, preload=True, verbose="ERROR")
        elif ext == ".fif":
            r = mne.io.read_raw_fif(p, preload=True, verbose="ERROR")
        else:
            raise ValueError(f"No reader configured for extension '{ext}' ({p})")
        raws.append(r)

    if len(raws) == 1:
        return raws[0]

    log.info(f"    concatenating {len(raws)} files")
    return mne.concatenate_raws(raws, verbose="ERROR")


def orient_montage(montage, flags: list[str]):
    """Rotate montage positions into MNE's RAS head frame (+X right, +Y anterior,
    +Z superior).

    ANT/ASA .elc files are commonly written ALS (+X anterior, +Y left, +Z up).
    Read as RAS that places frontal electrodes on the right and posterior ones
    on the left — a 90 degree rotation of the topography.

    Rather than assume ALS, the frame is inferred from anatomical landmarks:
    the anterior-minus-posterior vector must end up along +Y, and the
    right-minus-left vector along +X. The third axis is then the cross product
    of the first two, which guarantees a proper rotation — so this can correct
    an axis permutation but can never mirror left and right, which would be far
    worse than the original problem and much harder to notice.
    """
    import mne

    mode = CONFIG["MONTAGE_ORIENTATION"]
    if mode == "none":
        return montage, {"applied": False, "reason": "MONTAGE_ORIENTATION = none"}

    pos = montage.get_positions()
    ch_pos = {k: np.asarray(v, dtype=float) for k, v in pos["ch_pos"].items()
              if v is not None and np.all(np.isfinite(v))}
    if not ch_pos:
        return montage, {"applied": False, "reason": "no finite positions"}

    lut = {norm(k): k for k in ch_pos}

    def group_mean(names: list[str]) -> np.ndarray | None:
        hits = [ch_pos[lut[norm(n)]] for n in names if norm(n) in lut]
        return np.mean(hits, axis=0) if hits else None

    if mode == "als_to_ras":
        M = np.array([[0., -1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]])
        info = {"applied": True, "mode": "als_to_ras", "matrix": M.tolist()}
    else:
        lm = CONFIG["MONTAGE_LANDMARKS"]
        ant, post = group_mean(lm["anterior"]), group_mean(lm["posterior"])
        left, right = group_mean(lm["left"]), group_mean(lm["right"])

        if ant is None or post is None or left is None or right is None:
            missing = [k for k, v in [("anterior", ant), ("posterior", post),
                                      ("left", left), ("right", right)] if v is None]
            msg = (f"montage orientation not checked — no landmark electrodes for "
                   f"{missing}; positions left as written")
            flags.append(msg)
            log.warning(f"    {msg}")
            return montage, {"applied": False, "reason": f"missing landmarks: {missing}"}

        v_ap = ant - post           # must point +Y in RAS
        v_lr = right - left         # must point +X in RAS

        ax_ap = int(np.argmax(np.abs(v_ap)))
        ax_lr = int(np.argmax(np.abs(v_lr)))
        if ax_ap == ax_lr:
            msg = ("montage orientation ambiguous — anterior-posterior and "
                   "left-right resolve to the same axis; positions left as written")
            flags.append(msg)
            log.warning(f"    {msg}")
            return montage, {"applied": False, "reason": "degenerate axes"}

        ex = np.zeros(3); ex[ax_lr] = np.sign(v_lr[ax_lr])      # -> +X (right)
        ey = np.zeros(3); ey[ax_ap] = np.sign(v_ap[ax_ap])      # -> +Y (anterior)
        ez = np.cross(ex, ey)                                   # -> +Z, right-handed
        M = np.vstack([ex, ey, ez])

        if np.allclose(M, np.eye(3)):
            log.info("    montage already in RAS orientation — no rotation needed")
            return montage, {"applied": False, "reason": "already RAS",
                             "matrix": M.tolist()}

        info = {"applied": True, "mode": "auto", "matrix": M.tolist(),
                "detected_anterior_axis": ax_ap, "detected_leftright_axis": ax_lr}

    def rot(v):
        return None if v is None else (M @ np.asarray(v, dtype=float))

    new_ch = {k: M @ v for k, v in ch_pos.items()}
    montage_out = mne.channels.make_dig_montage(
        ch_pos=new_ch,
        nasion=rot(pos.get("nasion")), lpa=rot(pos.get("lpa")), rpa=rot(pos.get("rpa")),
        coord_frame="head",
    )

    # ---- verify the result actually looks like a head ----
    v = {k: None for k in ("anterior", "posterior", "left", "right")}
    npos = {k: M @ p for k, p in ch_pos.items()}
    lut2 = {norm(k): k for k in npos}

    def gm2(names):
        hits = [npos[lut2[norm(n)]] for n in names if norm(n) in lut2]
        return np.mean(hits, axis=0) if hits else None

    lm = CONFIG["MONTAGE_LANDMARKS"]
    a, p_, l, r = gm2(lm["anterior"]), gm2(lm["posterior"]), gm2(lm["left"]), gm2(lm["right"])
    checks = {}
    if a is not None and p_ is not None:
        checks["anterior_is_+Y"] = bool(a[1] > p_[1])
    if l is not None and r is not None:
        checks["right_is_+X"] = bool(r[0] > l[0])
    checks["mean_z_positive"] = bool(np.mean([q[2] for q in npos.values()]) > -0.05)
    info["verification"] = checks

    if not all(checks.values()):
        msg = f"montage rotation applied but verification failed: {checks}"
        flags.append(msg)
        log.error(f"    {msg}")
    else:
        log.info(f"    montage rotated into RAS ({info.get('mode')}) — "
                 f"anterior->+Y, right->+X, verified")
        flags.append("montage rotated into RAS orientation "
                     "(source file was not RAS)")

    return montage_out, info


def plot_montage_check(raw, out_png: Path, title: str) -> None:
    """Save a top-down sensor layout so the orientation can be eyeballed.

    Nose points up. Frontal labels must sit at the top, occipital at the bottom,
    T7 on the left, T8 on the right.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    try:
        raw.copy().pick("eeg").plot_sensors(
            show_names=True, axes=ax, show=False, sphere="auto")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"    could not plot montage: {exc}")
        plt.close(fig)
        return
    ax.set_title(f"{title}\nnose up · frontal top · T7 left · T8 right", fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    montage check figure: {out_png.name}")


def apply_montage(raw, elc_path: Path | None,
                  flags: list[str]) -> tuple[str, dict[str, Any]]:
    import mne

    if elc_path is None:
        flags.append("no .elc montage file found — positions not set")
        log.warning("    no .elc file found; continuing without montage")
        return "none", {"applied": False, "reason": "no montage file"}

    kwargs = {}
    if CONFIG["ELC_UNITS"] in ("mm", "m"):
        kwargs["coord_frame"] = "head"
    try:
        montage = mne.channels.read_custom_montage(elc_path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        # A malformed montage must not abort the file. Positions are only
        # strictly needed by ICLabel, which will report its own failure later.
        flags.append(f"could not read montage {elc_path.name}: "
                     f"{type(exc).__name__}: {exc}")
        log.error(f"    could not read montage {elc_path.name}: {exc}")
        log.error("    continuing WITHOUT channel positions — ICLabel will fail "
                  "if this is a full run")
        return (f"{elc_path.name} (FAILED TO READ)",
                {"applied": False, "reason": "montage unreadable"})

    montage, orient_info = orient_montage(montage, flags)

    raw_names = set(raw.ch_names)
    mon_names = set(montage.ch_names)

    if not CONFIG["MONTAGE_MATCH_CASE"]:
        lut = {norm(n): n for n in mon_names}
        rename = {r: lut[norm(r)] for r in raw_names if norm(r) in lut and lut[norm(r)] != r}
        if rename:
            log.info(f"    renaming {len(rename)} channels to match the montage")
            raw.rename_channels(rename)
            raw_names = set(raw.ch_names)

    # EOG is periocular and legitimately absent from a scalp .elc — only flag
    # EEG channels, where a missing position is a genuine problem.
    eeg_names = {ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"}
    missing = sorted(eeg_names - mon_names)
    if missing:
        flags.append(f"{len(missing)} EEG channel(s) absent from .elc: {missing[:8]}")
        log.warning(f"    {len(missing)} EEG channel(s) not in montage: {missing[:8]}")

    raw.set_montage(montage, on_missing="ignore", match_case=CONFIG["MONTAGE_MATCH_CASE"],
                    verbose="ERROR")
    log.info(f"    montage applied from {elc_path.name}")
    return elc_path.name, orient_info


def setup_channels(raw, flags: list[str]) -> tuple[list[str], list[str]]:
    """Retype EOG channels as 'eog' (kept), drop mastoids, drop anything else.

    Returns (eog_channel_names, dropped_channel_names).

    Retyping rather than dropping is what makes the EOG usable for ocular
    component detection later, while keeping it out of everything it must not
    influence: MNE's average reference, ASR and ICA all operate on data
    channels, and an 'eog'-typed channel is not one.
    """
    lut = {norm(c): c for c in raw.ch_names}

    # ---- EOG: retype, keep ----
    eog = [lut[norm(w)] for w in CONFIG["EOG_CHANNELS"] if norm(w) in lut]
    if eog:
        raw.set_channel_types({c: "eog" for c in eog}, verbose="ERROR")
        log.info(f"    EOG kept and retyped as 'eog': {eog}")
    else:
        flags.append("NO EOG channel matched the configured names — "
                     "ocular component detection will be skipped")
        log.warning("    no EOG channel matched CONFIG['EOG_CHANNELS'] — "
                    "EOG-based IC detection will be skipped")

    n_eog_expected = CONFIG["EXPECTED_N_EOG"]
    if len(eog) != n_eog_expected:
        flags.append(f"{len(eog)} EOG channel(s) found, expected {n_eog_expected}")

    # ---- Mastoids: drop ----
    to_drop = [lut[norm(w)] for w in CONFIG["MASTOID_CHANNELS"] if norm(w) in lut]
    to_drop = [c for c in to_drop if c not in eog]
    if to_drop:
        raw.drop_channels(to_drop)
        log.info(f"    dropped {len(to_drop)}: {to_drop}")
    else:
        flags.append("no mastoid channels matched the configured names")
        log.warning("    no mastoid channels matched CONFIG names — none dropped")

    # ---- Anything still not EEG or EOG would corrupt the average reference ----
    stray = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types())
             if t not in ("eeg", "eog")]
    if stray:
        flags.append(f"channels of unexpected type dropped: {stray}")
        log.warning(f"    dropping {len(stray)} channel(s) of unexpected type: {stray}")
        raw.drop_channels(stray)
        to_drop += stray

    types = raw.get_channel_types()
    n_eeg = sum(1 for t in types if t == "eeg")
    n_eog = sum(1 for t in types if t == "eog")
    expected = CONFIG["EXPECTED_N_EEG_AFTER_DROP"]
    if n_eeg != expected:
        flags.append(f"{n_eeg} EEG channels after drop, expected {expected}")
        log.warning(f"    {n_eeg} EEG channels retained, expected {expected}")
    log.info(f"    channels now: {n_eeg} EEG + {n_eog} EOG")

    return eog, to_drop



def apply_gedai_denoising(raw_filt, n_jobs=-1):
    from gedai import Gedai, AdaptiveMultibandGedai
    """
    Applies broadband GEDAI followed by adaptive multiband GEDAI 
    to an MNE Raw instance.
    ** THIS IS AN UPGRADED ASR THAT OUTPERFORMS PERVIOUS MODELS **
    Parameters:
    -----------
    raw_filt : mne.io.Raw
        The pre-filtered MNE Raw object to denoise.
    n_jobs : int
        Number of jobs to run in parallel (default: 1). Use -1 to use all cores.

    Returns:
    --------
    adaptive_multiband_denoised_raw : mne.io.Raw
        The fully denoised Raw object.
    """
    # 1. Apply Broadband GEDAI
    broadband_gedai = Gedai()
    broadband_gedai.fit_raw(raw_filt, noise_multiplier=6.0, n_jobs=n_jobs)

    broadband_denoised_raw = broadband_gedai.transform_raw(
        raw_filt,
        n_jobs=n_jobs,
        verbose=False
    )

    # 2. Apply Adaptive Multiband GEDAI
    adaptive_multiband_gedai = AdaptiveMultibandGedai(
        wavelet_type="haar",
        wavelet_level=5,
        cycles_per_wavelet=10
    )
    adaptive_multiband_gedai.fit_raw(
        broadband_denoised_raw,
        noise_multiplier=3.0,
        n_jobs=n_jobs
    )

    adaptive_multiband_denoised_raw = adaptive_multiband_gedai.transform_raw(
        broadband_denoised_raw,
        n_jobs=n_jobs,
        verbose=False
    )

    return adaptive_multiband_denoised_raw

def filter_chain(raw, flags: list[str]) -> None:
    from scipy import signal as sps

    sf_in = raw.info["sfreq"]
    target = CONFIG["RESAMPLE_HZ"]
    if abs(sf_in - target) > 1e-6:
        log.info(f"    resampling {sf_in:.1f} -> {target:.1f} Hz")
        raw.resample(target, npad="auto", verbose="ERROR")
    else:
        log.info(f"    already at {target:.1f} Hz, no resampling")

    log.info(f"    detrending ({CONFIG['DETREND_TYPE']})")
    raw.apply_function(
        lambda x: sps.detrend(x, type=CONFIG["DETREND_TYPE"]),
        picks="eeg", channel_wise=True, verbose="ERROR",
    )

    l_freq, h_freq = CONFIG["L_FREQ"], CONFIG["H_FREQ"]
    nyq = raw.info["sfreq"] / 2.0
    if h_freq >= nyq:
        h_freq = None
        flags.append(f"h_freq >= Nyquist ({nyq} Hz); low-pass skipped")
    log.info(f"    band-pass {l_freq}-{h_freq} Hz (zero-phase FIR, no notch)")
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir",
               phase=CONFIG["FILTER_PHASE"], fir_design=CONFIG["FIR_DESIGN"],
               verbose="ERROR")

    if CONFIG["APPLY_NOTCH"]:
        freqs = [f for f in CONFIG["NOTCH_FREQS"] if f < nyq]
        log.info(f"    notch at {freqs} Hz")
        raw.notch_filter(freqs, verbose="ERROR")

    log.info("    re-referencing to common average")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")


def eog_coupling(raw) -> float:
    """Strongest absolute correlation between the EOG channel and any EEG channel.

    A direct measure of how much ocular signal is present in the EEG. Computed
    before and after ASR: ASR removes high-variance transients, and blinks are
    exactly that, so an aggressive ASR pass can strip the ocular artifact out of
    the EEG before ICA ever sees it. If that happens, neither ICLabel nor the
    EOG channel can identify an ocular component — because there no longer is
    one. A large drop here explains an unexpectedly low ocular IC count.
    """
    types = raw.get_channel_types()
    if "eog" not in types:
        return float("nan")
    try:
        eog = raw.get_data(picks="eog")[0]
        eeg = raw.get_data(picks="eeg")
        eog = eog - eog.mean()
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
        denom = np.sqrt((eeg ** 2).sum(axis=1) * (eog ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (eeg @ eog) / denom
        return float(np.nanmax(np.abs(r)))
    except Exception:  # noqa: BLE001
        return float("nan")


def run_asr(raw, flags: list[str]) -> float:
    """Applies GEDAI denoising instead of standard ASR.
    Returns the fraction of samples changed."""
    if not CONFIG["ASR_ENABLED"]:
        return float("nan")

    try:
        from gedai import Gedai, AdaptiveMultibandGedai
    except ImportError:
        flags.append("gedai not installed — Denoising SKIPPED")
        log.error("    gedai not installed; Denoising skipped (pip install gedai)")
        return float("nan")

    log.info("    fitting ASR (Broadband + Adaptive Multiband)")

    data = raw.get_data(picks="eeg")

    denoised_raw = apply_gedai_denoising(raw.copy(), n_jobs=-1)


    cleaned = denoised_raw.get_data(picks="eeg")


    if cleaned.shape != data.shape:
        n = min(cleaned.shape[1], data.shape[1])
        cleaned, data = cleaned[:, :n], data[:, :n]
        flags.append("GEDAI returned a different sample count; data truncated to match")


    changed = np.abs(cleaned - data) > (1e-3 * (np.std(data) + 1e-24))
    pct = float(np.mean(np.any(changed, axis=0)))
    log.info(f"    GEDAI modified {pct * 100:.1f}% of samples")

    # Inject the cleaned data back into the original raw object in-place
    picks = [raw.ch_names.index(c) for c in raw.copy().pick("eeg").ch_names]
    raw._data[picks, :cleaned.shape[1]] = cleaned

    return pct


def run_ica(raw, flags: list[str]) -> tuple[Any, dict[str, Any]]:
    """Fit extended-Infomax ICA, classify with ICLabel, reject non-brain ICs."""
    import mne
    from mne.preprocessing import ICA

    n_eeg = sum(1 for t in raw.get_channel_types() if t == "eeg")
    rank_dict = mne.compute_rank(raw, rank=None, verbose="ERROR")
    rank_est = int(rank_dict.get("eeg", n_eeg))
    n_comp = int(min(CONFIG["ICA_N_COMPONENTS"], rank_est))
    if n_comp < CONFIG["ICA_N_COMPONENTS"]:
        flags.append(f"rank {rank_est} < {CONFIG['ICA_N_COMPONENTS']}; "
                     f"ICA fitted with {n_comp} components")
        log.warning(f"    estimated rank {rank_est}; fitting {n_comp} components")
    else:
        log.info(f"    estimated rank {rank_est}; fitting {n_comp} components")

    ica = ICA(
        n_components=n_comp,
        method=CONFIG["ICA_METHOD"],
        fit_params=dict(extended=True) if CONFIG["ICA_EXTENDED"] else None,
        random_state=CONFIG["ICA_RANDOM_STATE"],
        max_iter=CONFIG["ICA_MAX_ITER"],
    )

    # ICLabel expects 1-100 Hz data; fit on a high-passed copy, apply to the
    # 0.1 Hz data. The analysed frequency band is unaffected.
    if CONFIG["ICA_FIT_HPF"] is not None:
        log.info(f"    fitting ICA on a {CONFIG['ICA_FIT_HPF']} Hz high-passed copy")
        raw_fit = raw.copy().filter(
            l_freq=CONFIG["ICA_FIT_HPF"], h_freq=None, method="fir",
            phase=CONFIG["FILTER_PHASE"], fir_design=CONFIG["FIR_DESIGN"],
            verbose="ERROR",
        )
    else:
        raw_fit = raw

    ica.fit(raw_fit, verbose="ERROR")

    try:
        from mne_icalabel import label_components
        res = label_components(raw_fit, ica, method="iclabel")
        labels = list(res["labels"])
        probs = np.asarray(res["y_pred_proba"], dtype=float).ravel().tolist()
    except ImportError:
        flags.append("mne-icalabel not installed — NO components rejected")
        log.error("    mne-icalabel not installed (pip install mne-icalabel)")
        labels = ["brain"] * n_comp
        probs = [float("nan")] * n_comp

    keep = set(CONFIG["KEEP_LABELS"])
    thr = CONFIG["ICLABEL_PROB_THRESHOLD"]
    excl_iclabel = [
        i for i, (lab, p) in enumerate(zip(labels, probs))
        if lab not in keep and (np.isnan(p) or p >= thr)
    ]

    # ---- EOG-channel-based detection, in addition to ICLabel ----
    excl_eog: list[int] = []
    eog_scores: list[float] = []
    eog_names = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eog"]

    if CONFIG["EOG_DETECT_ENABLED"] and eog_names:
        try:
            idx, scores = ica.find_bads_eog(
                raw_fit,
                threshold=CONFIG["EOG_DETECT_THRESHOLD"],
                measure=CONFIG["EOG_DETECT_MEASURE"],
                verbose="ERROR",
            )
            excl_eog = sorted(int(i) for i in idx)
            scores = np.atleast_2d(np.asarray(scores, dtype=float))
            # One row per EOG channel; keep the strongest association per IC.
            eog_scores = np.abs(scores).max(axis=0).tolist()
            log.info(f"    EOG detection on {eog_names} "
                     f"({CONFIG['EOG_DETECT_MEASURE']} > {CONFIG['EOG_DETECT_THRESHOLD']}): "
                     f"{len(excl_eog)} component(s) {excl_eog}")
        except Exception as exc:  # noqa: BLE001
            flags.append(f"EOG component detection failed: {type(exc).__name__}: {exc}")
            log.error(f"    EOG component detection failed: {exc}")
    elif CONFIG["EOG_DETECT_ENABLED"]:
        log.warning("    no EOG channel present — EOG component detection skipped")

    # Union: EOG detection can only add components, never remove any.
    exclude = sorted(set(excl_iclabel) | set(excl_eog))
    eog_only = sorted(set(excl_eog) - set(excl_iclabel))
    ica.exclude = exclude

    # What the more permissive convention would have given — logged for contrast.
    keep_alt = keep | {"other"}
    n_excl_alt = len(set(i for i, lab in enumerate(labels) if lab not in keep_alt)
                     | set(excl_eog))

    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1

    detail = {
        "rank_est": rank_est,
        "n_components": n_comp,
        "method": CONFIG["ICA_METHOD"],
        "extended": CONFIG["ICA_EXTENDED"],
        "random_state": CONFIG["ICA_RANDOM_STATE"],
        "fit_highpass_hz": CONFIG["ICA_FIT_HPF"],
        "keep_labels": sorted(keep),
        "label_counts": counts,
        "eog_detection": {
            "enabled": CONFIG["EOG_DETECT_ENABLED"],
            "eog_channels": eog_names,
            "measure": CONFIG["EOG_DETECT_MEASURE"],
            "threshold": CONFIG["EOG_DETECT_THRESHOLD"],
            "rejected_indices": excl_eog,
            "rejected_not_caught_by_iclabel": eog_only,
        },
        "components": [
            {"index": i, "label": lab,
             "probability": None if np.isnan(p) else round(p, 4),
             "eog_score": (round(eog_scores[i], 4)
                           if i < len(eog_scores) and np.isfinite(eog_scores[i]) else None),
             "rejected_by_iclabel": i in excl_iclabel,
             "rejected_by_eog": i in excl_eog,
             "rejected": i in exclude}
            for i, (lab, p) in enumerate(zip(labels, probs))
        ],
        "n_rejected_iclabel": len(excl_iclabel),
        "n_rejected_eog": len(excl_eog),
        "n_rejected_eog_only": len(eog_only),
        "n_rejected": len(exclude),
        "pct_rejected": round(len(exclude) / n_comp, 4) if n_comp else float("nan"),
        "n_rejected_if_other_kept": n_excl_alt,
        "rejected_indices": exclude,
    }

    log.info(f"    ICLabel: {counts}")
    log.info(f"    rejecting {len(exclude)}/{n_comp} components "
             f"({100 * len(exclude) / max(n_comp, 1):.1f}%) — "
             f"ICLabel {len(excl_iclabel)}, EOG {len(excl_eog)}, "
             f"EOG-only {len(eog_only)} "
             f"[{n_excl_alt} if 'other' were kept]")
    if eog_only:
        log.info(f"    EOG caught {len(eog_only)} component(s) ICLabel did not: "
                 f"{[(i, labels[i]) for i in eog_only]}")

    if CONFIG["ICA_FIT_HPF"] is not None:
        del raw_fit

    return ica, detail


# ==============================================================================
# Per-file driver
# ==============================================================================

def process_recording(root: Path, deriv: Path, subject: str, rec_key: str,
                      elc_path: Path | None, inventory_only: bool = False,
                      plot_montage: bool = False) -> dict[str, Any]:
    import mne
    from scipy import signal as sps
    spec = CONFIG["RECORDINGS"][rec_key]
    kind = spec["kind"]
    flags: list[str] = []
    row: dict[str, Any] = {
        "subject": subject, "recording": rec_key, "kind": kind,
        "status": "pending", "timestamp_utc": utcnow(),
    }

    paths = find_recording_files(root, subject, rec_key)
    if not paths:
        row.update(status="failed", error="no input file found", flags="missing input")
        log.error(f"  [{subject}/{rec_key}] no input file found")
        return row

    row["n_input_files"] = len(paths)
    row["input_files"] = " | ".join(p.name for p in paths)
    log.info(f"  [{subject}/{rec_key}] {len(paths)} file(s)")

    raw = load_raw(paths)
    row["sfreq_in"] = round(float(raw.info["sfreq"]), 3)
    row["duration_s"] = round(float(raw.n_times / raw.info["sfreq"]), 2)
    row["n_chan_in"] = len(raw.ch_names)
    log.info(f"    {row['n_chan_in']} channels, {row['duration_s']:.1f}s @ {row['sfreq_in']} Hz")

    # Drop BEFORE applying the montage. EOG has no scalp position, so montaging
    # first produces a "not in montage" warning for a channel that is discarded
    # one step later — noise in the flags column that would mask a genuine
    # montage mismatch. Dropping first makes any remaining warning meaningful.
    eog_names, dropped = setup_channels(raw, flags)
    montage_src, orient_info = apply_montage(raw, elc_path, flags)
    types = raw.get_channel_types()
    row["n_eeg_kept"] = sum(1 for t in types if t == "eeg")
    row["n_eog_kept"] = len(eog_names)
    row["eog_channels"] = " | ".join(eog_names)
    row["dropped_channels"] = " | ".join(dropped)

    if plot_montage:
        plot_montage_check(raw, deriv / subject / f"{subject}_{rec_key}_montage.png",
                           f"{subject} / {rec_key} — montage as applied")

    # ---- Edge crop and break removal, BEFORE any filtering ----
    row["duration_raw_s"] = row["duration_s"]

    edge_rep = crop_to_markers(raw, kind, flags)
    row["crop_start_s"] = edge_rep.get("crop_start_s")
    row["crop_end_s"] = edge_rep.get("crop_end_s")
    row["duration_after_edge_crop_s"] = round(raw.n_times / raw.info["sfreq"], 2)

    breaks, break_rep = find_breaks(raw, kind, flags)
    gs = break_rep.get("gap_stats") or {}
    row["gap_p50_s"], row["gap_p95_s"] = gs.get("p50"), gs.get("p95")
    row["gap_max_s"] = gs.get("max")

    raw, drop_rep = drop_breaks(raw, breaks, flags)
    row["n_breaks_dropped"] = break_rep.get("n_dropped", 0)
    row["break_total_s"] = break_rep.get("total_dropped_s", 0.0)
    row["duration_after_breaks_s"] = round(raw.n_times / raw.info["sfreq"], 2)
    row["duration_s"] = row["duration_after_breaks_s"]

    if row["n_breaks_dropped"]:
        flags.append(f"{row['n_breaks_dropped']} break period(s) removed, "
                     f"{row['break_total_s']:.0f}s total")

    inv = inventory_annotations(raw, kind)
    flags.extend(inv["flags"])
    for code in CONFIG["REST_CODES"]:
        row[f"n_ann_{code}"] = inv.get(f"n_ann_{code}", 0)
    row["n_blocks_valid"] = inv["n_blocks_valid"]
    row["n_blocks_short"] = inv["n_blocks_short"]
    log.info(f"    annotations: {inv['n_annotations_total']} total, "
             f"{inv['n_blocks_valid']} valid block(s), {inv['n_blocks_short']} short")

    record: dict[str, Any] = {
        "subject": subject,
        "recording": rec_key,
        "kind": kind,
        "timestamp_utc": utcnow(),
        "input_files": [str(p) for p in paths],
        "montage_source": montage_src,
        "montage_orientation": orient_info,
        "channels_dropped": dropped,
        "eog_channels_kept": eog_names,
        "channel_names_after_drop": list(raw.ch_names),
        "channel_types_after_drop": list(raw.get_channel_types()),
        "sfreq_in": row["sfreq_in"],
        "duration_s": row["duration_s"],
        "edge_crop": edge_rep,
        "break_detection": break_rep,
        "break_removal": drop_rep,
        "annotation_inventory": inv,
        "config": {k: v for k, v in CONFIG.items() if k != "RECORDINGS"},
        "versions": {"mne": mne.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
    }

    if inventory_only:
        row["status"] = "inventory"
        row["flags"] = "; ".join(flags)
        record["mode"] = "inventory_only"
        write_json(deriv / subject / f"{subject}_{rec_key}_inventory.json", record)
        return row

    filter_chain(raw, flags)
    row["sfreq_out"] = round(float(raw.info["sfreq"]), 3)

    coup_before = eog_coupling(raw)
    asr_pct = run_asr(raw, flags)
    coup_after = eog_coupling(raw)
    row["asr_pct_reconstructed"] = None if np.isnan(asr_pct) else round(asr_pct, 4)
    row["eog_eeg_corr_pre_asr"] = None if np.isnan(coup_before) else round(coup_before, 4)
    row["eog_eeg_corr_post_asr"] = None if np.isnan(coup_after) else round(coup_after, 4)

    if np.isfinite(coup_before) and np.isfinite(coup_after):
        log.info(f"    EOG-EEG coupling: {coup_before:.3f} -> {coup_after:.3f} across ASR")
        if coup_before > 0.3 and coup_after < 0.5 * coup_before:
            msg = (f"ASR removed most ocular signal from the EEG "
                   f"(EOG-EEG correlation {coup_before:.2f} -> {coup_after:.2f}); "
                   f"expect few or no ocular ICs to be identifiable")
            flags.append(msg)
            log.warning(f"    {msg}")

    record["asr"] = {"enabled": CONFIG["ASR_ENABLED"], "cutoff": CONFIG["ASR_CUTOFF"],
                     "method": CONFIG["ASR_METHOD"],
                     "calibration_highpass_hz": CONFIG["ASR_CALIB_HPF"],
                     "pct_samples_reconstructed": row["asr_pct_reconstructed"],
                     "eog_eeg_corr_pre_asr": row["eog_eeg_corr_pre_asr"],
                     "eog_eeg_corr_post_asr": row["eog_eeg_corr_post_asr"]}

    ica, ica_detail = run_ica(raw, flags)
    record["ica"] = ica_detail
    row["rank_est"] = ica_detail["rank_est"]
    row["n_ica"] = ica_detail["n_components"]
    row["n_ic_rejected"] = ica_detail["n_rejected"]
    row["pct_ic_rejected"] = ica_detail["pct_rejected"]
    row["n_ic_iclabel"] = ica_detail["n_rejected_iclabel"]
    row["n_ic_eog"] = ica_detail["n_rejected_eog"]
    row["n_ic_eog_only"] = ica_detail["n_rejected_eog_only"]

    # ---- QC gate ----
    if ica_detail["pct_rejected"] > reject_ic_threshold: # CONFIG["MAX_IC_REJECT_FRACTION"]:
        msg = (f"{ica_detail['n_rejected']}/{ica_detail['n_components']} ICs rejected "
               f"(>= {CONFIG['MAX_IC_REJECT_FRACTION']:.0%}) — marked noisy")
        flags.append(msg)
        log.warning(f"    EXCLUDED: {msg}")
        row["status"] = "excluded_noisy"
        row["flags"] = "; ".join(flags)
        record["excluded"] = {"reason": msg}
        write_json(deriv / subject / f"{subject}_{rec_key}_stage0.json", record)
        return row

    log.info("    applying ICA and reconstructing")
    ica.apply(raw, verbose="ERROR")
    raw.apply_function(
        lambda x: sps.detrend(x, type=CONFIG["DETREND_TYPE"]),
        picks="eeg", channel_wise=True, verbose="ERROR",
    )

    if CONFIG["DROP_EOG_AFTER_ICA"] and eog_names:
        raw.drop_channels(eog_names)
        log.info(f"    dropped EOG after ICA: {eog_names}")
        record["eog_dropped_after_ica"] = eog_names

    out_dir = deriv / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / f"{subject}_{rec_key}_clean-raw.fif"
    raw.save(raw_out, overwrite=CONFIG["FIF_OVERWRITE"], verbose="ERROR")
    record["output_raw"] = str(raw_out)
    log.info(f"    saved {raw_out.name}")

    if CONFIG["SAVE_ICA"]:
        ica_out = out_dir / f"{subject}_{rec_key}-ica.fif"
        ica.save(ica_out, overwrite=CONFIG["FIF_OVERWRITE"], verbose="ERROR")
        record["output_ica"] = str(ica_out)

    write_json(out_dir / f"{subject}_{rec_key}_stage0.json", record)
    row["status"] = "done"
    row["flags"] = "; ".join(flags)
    return row


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ==============================================================================
# Manifest
# ==============================================================================

# The manifest is held as a plain list of dicts, NOT a DataFrame.
#
# Manifest cells hold mixed types — strings, ints, floats, None. A DataFrame
# infers a dtype per column, so a column that is all-NaN on the first pass gets
# float64, and later writing a filename into it raises:
#
#     TypeError: Invalid value 'hx_01_post_task_rest.vhdr' for dtype 'float64'
#     pandas.errors.LossySetitemError
#
# on pandas >= 2.2, where silent upcasting was removed. Dicts have no dtype, so
# the problem cannot occur. pandas is used only to write the CSV at the end.

def blank_row(subject: str, recording: str, kind: str, status: str) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in MANIFEST_COLUMNS}
    row.update(subject=subject, recording=recording, kind=kind,
               status=status, timestamp_utc=utcnow())
    return row


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Read an existing manifest back as plain dicts. Everything stays a string;
    the values are only ever compared or re-written, never computed with."""
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict("records"):
        row: dict[str, Any] = {c: None for c in MANIFEST_COLUMNS}
        for k, v in rec.items():
            if k in row:
                row[k] = v if v not in ("", "nan") else None
        rows.append(row)
    return rows


def build_manifest(manifest: list[dict[str, Any]],
                   subjects: list[str]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {
        (str(r.get("subject")), str(r.get("recording"))): r for r in manifest
    }
    for s in subjects:
        for rec_key, spec in CONFIG["RECORDINGS"].items():
            if (s, rec_key) not in index:
                status = ("excluded_by_request" if s in CONFIG["EXCLUDE_SUBJECTS"]
                          else "pending")
                index[(s, rec_key)] = blank_row(s, rec_key, spec["kind"], status)

    # Excluded subjects always win, even if a previous run processed them.
    for (s, _), row in index.items():
        if s in CONFIG["EXCLUDE_SUBJECTS"]:
            row["status"] = "excluded_by_request"

    return [index[k] for k in sorted(index)]


def save_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(path, index=False)


# ==============================================================================
# Summary report
# ==============================================================================

def write_summary(df: pd.DataFrame, path: Path) -> None:
    counts = df["status"].value_counts().to_dict()
    flagged = df[df["flags"].notna() & (df["flags"].astype(str).str.len() > 0)]

    def table(d: pd.DataFrame, cols: list[str]) -> str:
        cols = [c for c in cols if c in d.columns]
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join(
            "<tr>" + "".join(f"<td>{'' if pd.isna(r[c]) else r[c]}</td>" for c in cols) + "</tr>"
            for _, r in d.iterrows()
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stage 0 — QC summary</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
margin:0;background:#faf9f7;color:#1a1a1a;line-height:1.55}}
.wrap{{max-width:1200px;margin:0 auto;padding:40px 28px 80px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:36px 0 8px;
padding-top:12px;border-top:1px solid #e0ddd8}}
.sub{{color:#5c5c5c;font-size:14px;margin:0 0 20px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff;margin:10px 0}}
th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid #e0ddd8;vertical-align:top}}
th{{background:#f2efea;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#5c5c5c}}
.pill{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;
font-weight:600;margin-right:8px}}
</style></head><body><div class="wrap">
<h1>Stage 0 — QC summary</h1>
<p class="sub">Generated {utcnow()} · {len(df)} recording files</p>
<div>{"".join(f'<span class="pill" style="background:#eef3f6">{k}: {v}</span>'
              for k, v in counts.items())}</div>
<h2>All recordings</h2>
{table(df, ["subject","recording","status","duration_raw_s","duration_after_edge_crop_s",
            "n_breaks_dropped","break_total_s","duration_s",
            "gap_p50_s","gap_p95_s","gap_max_s",
            "sfreq_in","n_eeg_kept","n_eog_kept",
            "rank_est","n_ica","n_ic_rejected","pct_ic_rejected",
            "n_ic_iclabel","n_ic_eog","n_ic_eog_only",
            "asr_pct_reconstructed","eog_eeg_corr_pre_asr","eog_eeg_corr_post_asr",
            "n_ann_1002","n_ann_1004",
            "n_blocks_valid","n_blocks_short"])}
<h2>Flagged for review ({len(flagged)})</h2>
{table(flagged, ["subject","recording","status","flags","error"]) if len(flagged)
 else "<p class='sub'>Nothing flagged.</p>"}
</div></body></html>"""
    path.write_text(html, encoding="utf-8")


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 0 — shared EEG preprocessing (BeechBand study)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", required=True, type=Path, help="main_folder containing hx_XX/")
    ap.add_argument("--derivatives", type=Path, default=None,
                    help="output folder (default: <root>/derivatives)")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="process only these subjects, e.g. hx_04 hx_07")
    ap.add_argument("--recordings", nargs="*", default=None,
                    choices=list(CONFIG["RECORDINGS"]), help="process only these recordings")
    ap.add_argument("--force", action="store_true", help="reprocess rows already marked done")
    ap.add_argument("--inventory", action="store_true",
                    help="read files and report channels/annotations only — no processing")
    ap.add_argument("--plot-montage", action="store_true",
                    help="save a top-down sensor layout PNG per recording so the "
                         "montage orientation can be checked visually")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
        return 2

    deriv: Path = (args.derivatives or root / CONFIG["DERIVATIVES_DIRNAME"]).expanduser().resolve()
    deriv.mkdir(parents=True, exist_ok=True)
    setup_logging(deriv / "stage0_run.log", verbose=not args.quiet)

    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        log.error("MNE is not installed. pip install 'mne>=1.6' mne-icalabel meegkit")
        return 2

    mode = "INVENTORY (no processing)" if args.inventory else "FULL"
    log.info("=" * 78)
    log.info(f"STAGE 0 — {mode}")
    log.info(f"root        : {root}")
    log.info(f"derivatives : {deriv}")
    log.info(f"mne         : {mne.__version__}")
    log.info("=" * 78)

    subjects = args.subjects or find_subjects(root)
    if not subjects:
        log.error(f"No subject folders matching {CONFIG['SUBJECT_PATTERN']} in {root}")
        return 1
    log.info(f"{len(subjects)} subject folder(s) found")

    elc = find_elc(root)
    log.info(f"montage     : {elc.name if elc else 'NOT FOUND'}")

    manifest_path = deriv / ("stage0_inventory_manifest.csv" if args.inventory
                             else "stage0_manifest.csv")
    manifest = build_manifest(load_manifest(manifest_path), subjects)
    save_manifest(manifest, manifest_path)

    rec_keys = args.recordings or list(CONFIG["RECORDINGS"])
    skip_states = {"done", "excluded_by_request", "excluded_noisy"}
    n_done = n_skip = n_fail = 0

    for r in manifest:
        sub, rec = str(r["subject"]), str(r["recording"])
        if sub not in subjects or rec not in rec_keys:
            continue
        if sub in CONFIG["EXCLUDE_SUBJECTS"]:
            log.info(f"[{sub}/{rec}] skipped — excluded_by_request")
            n_skip += 1
            continue
        if not args.inventory and not args.force and str(r["status"]) in skip_states:
            log.info(f"[{sub}/{rec}] skipped — status={r['status']}")
            n_skip += 1
            continue

        try:
            row = process_recording(root, deriv, sub, rec, elc,
                                    inventory_only=args.inventory,
                                    plot_montage=args.plot_montage)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error(f"[{sub}/{rec}] FAILED: {exc}")
            log.debug(tb)
            row = {"subject": sub, "recording": rec, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}", "timestamp_utc": utcnow()}
            (deriv / sub).mkdir(parents=True, exist_ok=True)
            (deriv / sub / f"{sub}_{rec}_ERROR.txt").write_text(tb, encoding="utf-8")

        # r is the dict living inside `manifest`, so updating it in place
        # updates the manifest. No dtype coercion involved.
        r.update({k: v for k, v in row.items() if k in MANIFEST_COLUMNS})
        # Flush after every file so an interruption never loses progress.
        save_manifest(manifest, manifest_path)

        if row["status"] in ("done", "inventory"):
            n_done += 1
        elif row["status"] == "failed":
            n_fail += 1
        else:
            n_skip += 1

    report = deriv / ("stage0_inventory_report.html" if args.inventory
                      else "stage0_summary_report.html")
    write_summary(pd.DataFrame(manifest, columns=MANIFEST_COLUMNS), report)

    log.info("=" * 78)
    log.info(f"processed {n_done} · skipped {n_skip} · failed {n_fail}")
    log.info(f"manifest : {manifest_path}")
    log.info(f"report   : {report}")
    log.info("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
