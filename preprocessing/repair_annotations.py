#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANNOTATION MARKER REPAIR
BeechBand vibrotactile stimulation study
================================================================================

Standalone utility, independent of stage0_preprocess.py. Runs BEFORE Stage 0,
on resting-state recordings only.

Does exactly one thing: fixes mis-typed annotation codes.

Some blocks were stamped with the wrong code at one boundary — e.g. onset
"1004" but offset "1003". The pairs are aliases for the same condition, so the
alias is rewritten onto the canonical code:

    1003  ->  1004     eyes closed, STIM-ON
    1001  ->  1002     eyes closed, STIM-OFF

The EEG signal is not touched. No cropping, no resampling, no filtering, no
change to block durations. Only the annotation descriptions change.

Blocks are detected and reported afterwards purely so you can confirm the
repair produced the expected 4 blocks (2 ON, 2 OFF) per file. Nothing is
dropped or altered on the basis of that report.

Output goes to a NEW directory tree that mirrors the source layout, so the raw
data is never modified:

    main_folder/                     <- untouched
    main_folder/updated/             <- created by this script
        waveguard64.elc              <- montage copied across
        hx_01/rest/hx_01_pre_task_rest_fixed.vhdr / .eeg / .vmrk
        hx_01/rest/hx_01_post_task_rest_fixed.vhdr / ...
        derivatives/repair_annotations_manifest.csv, repair_annotations_run.log

Because the layout mirrors the source, Stage 0 can be pointed straight at it:

    python stage0_preprocess.py --root main_folder/updated

Add --copy-task to carry the untouched task recordings across as well, so the
new tree is fully self-contained.

Usage
-----
  # Report what would change — writes nothing. Run this first.
  python repair_annotations.py --root /path/to/main_folder --dry-run

  # Do it
  python repair_annotations.py --root /path/to/main_folder

  # Self-contained new tree
  python repair_annotations.py --root /path/to/main_folder --copy-task

  # Lossless output instead of BrainVision
  python repair_annotations.py --root /path/to/main_folder --format fif

Requirements
------------
  pip install "mne>=1.6" pybv numpy pandas
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG: dict[str, Any] = {

    # ---- Subjects & layout --------------------------------------------------
    "EXCLUDE_SUBJECTS": [],
    "SUBJECT_PATTERN": r"^hx_\d+$",
    "REST_RECORDINGS": {
        "pre_task_rest":  {"subdir": "rest", "stem": "pre_task_rest"},
        "post_task_rest": {"subdir": "rest", "stem": "post_task_rest"},
    },
    "RAW_EXTENSIONS": [".vhdr"],

    # ---- The repair ---------------------------------------------------------
    # alias code -> canonical code
    "CODE_ALIASES": {"1003": "1004", "1001": "1002"},
    "CANONICAL_CODES": {"1002": "eyes_closed_stim_off", "1004": "eyes_closed_stim_on"},

    # ---- Post-repair validation (REPORT ONLY — nothing is dropped) ----------
    "MIN_BLOCK_DURATION_S": 59.0,
    "EXPECTED_BLOCKS_PER_CODE": 2,

    # ---- Output -------------------------------------------------------------
    "OUTPUT_ROOT_NAME": "updated",   # created under --root unless --output-root given
    "OUTPUT_SUBDIR": "rest",         # mirrors main_folder/hx_XX/rest/
    "OUTPUT_SUFFIX": "",
    "OUTPUT_FORMAT": "brainvision",  # "brainvision" (.vhdr) | "fif" (lossless)
    "OVERWRITE": True,
    "COPY_TASK_FILES": False,
    "TASK_SUBDIR": "task",
}

MANIFEST_COLUMNS = [
    "subject", "recording", "status",
    "input_file", "output_file",
    "duration_s", "sfreq", "n_ann_total",
    "n_renamed_1003_to_1004", "n_renamed_1001_to_1002", "n_renamed_total",
    "n_ann_1002_after", "n_ann_1004_after",
    "n_blocks_valid", "n_blocks_short", "block_durations_s",
    "flags", "error", "timestamp_utc",
]

log = logging.getLogger("repair")


# ==============================================================================
# Helpers
# ==============================================================================

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(logfile: Path, verbose: bool = True) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.propagate = False
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
    log.addHandler(sh)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract_code(description: str) -> str | None:
    """Pull the trigger code out of an annotation description.

    Handles bare '1004', 'Stimulus/S1004', 'Comment/1004', 'Trigger 1004'.
    """
    m = re.findall(r"\d+", str(description))
    return m[-1] if m else None


def find_subjects(root: Path) -> list[str]:
    pat = re.compile(CONFIG["SUBJECT_PATTERN"])
    return sorted(p.name for p in root.iterdir() if p.is_dir() and pat.match(p.name))


def find_recording_files(root: Path, subject: str, rec_key: str) -> list[Path]:
    spec = CONFIG["REST_RECORDINGS"][rec_key]
    sub_dir = root / subject
    stem_n = norm(spec["stem"])
    other = [norm(v["stem"]) for k, v in CONFIG["REST_RECORDINGS"].items() if k != rec_key]
    suffix_n = norm(CONFIG["OUTPUT_SUFFIX"])

    for ext in CONFIG["RAW_EXTENSIONS"]:
        hits: list[Path] = []
        for d in (sub_dir / spec["subdir"], sub_dir):
            if not d.is_dir():
                continue
            for f in sorted(d.glob(f"*{ext}")):
                fn = norm(f.stem)
                if stem_n not in fn:
                    continue
                if suffix_n and fn.endswith(suffix_n):
                    continue                      # never re-process our own output
                if any(o in fn for o in other if len(o) > len(stem_n)):
                    continue
                hits.append(f)
            if hits:
                break
        if hits:
            return hits
    return []


def load_raw(paths: list[Path]):
    import mne
    raws = []
    for p in paths:
        ext = p.suffix.lower()
        log.info(f"    reading {p.name}")
        if ext == ".cnt":
            try:
                r = mne.io.read_raw_ant(p, preload=True, verbose="ERROR")
            except Exception:  # noqa: BLE001
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
            raise ValueError(f"No reader configured for '{ext}' ({p})")
        raws.append(r)
    return raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, verbose="ERROR")


# ==============================================================================
# The repair
# ==============================================================================

def repair_annotations(raw, flags: list[str]) -> dict[str, Any]:
    """Rewrite alias codes onto their canonical code, in place.

    Only the numeric run inside the description is replaced; surrounding text is
    preserved, so 'Stimulus/S1003' becomes 'Stimulus/S1004'. Onsets and
    durations are untouched, as is the EEG signal.
    """
    import mne

    ann = raw.annotations
    descs = [str(d) for d in ann.description]
    aliases = CONFIG["CODE_ALIASES"]

    counts = {f"{a}->{c}": 0 for a, c in aliases.items()}
    changes: list[dict[str, Any]] = []
    new_descs: list[str] = []

    for i, d in enumerate(descs):
        code = extract_code(d)
        if code in aliases:
            canon = aliases[code]
            idx = d.rfind(code)                      # last numeric run only
            nd = d[:idx] + canon + d[idx + len(code):]
            counts[f"{code}->{canon}"] += 1
            changes.append({
                "index": i,
                "onset_s": round(float(ann.onset[i]), 3),
                "from": d, "to": nd,
            })
            new_descs.append(nd)
        else:
            new_descs.append(d)

    total = sum(counts.values())
    if total:
        raw.set_annotations(mne.Annotations(
            onset=ann.onset, duration=ann.duration,
            description=new_descs, orig_time=ann.orig_time))
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items() if v)
        log.info(f"    repaired {total} marker(s) — {summary}")
        for c in changes:
            log.debug(f"      @{c['onset_s']:8.2f}s   {c['from']!r} -> {c['to']!r}")
        flags.append(f"renamed {total} marker(s): {summary}")
    else:
        log.info("    no alias codes present — nothing to repair")

    return {"counts": counts, "total": total, "changes": changes}


# ==============================================================================
# Post-repair validation — REPORT ONLY
# ==============================================================================

@dataclass
class Block:
    code: str
    label: str
    onset: float
    offset: float
    duration: float
    valid: bool
    reason: str = ""


def detect_blocks(raw) -> tuple[list[Block], list[Block], list[str]]:
    """Report the blocks implied by the repaired markers.

    A candidate block is the interval between two consecutive annotations of the
    same code that contains no annotation of the other code. Candidates at or
    above the minimum duration are counted as valid, resolving overlaps
    earliest-first. Purely diagnostic — nothing is dropped or modified.
    """
    descs = [str(d) for d in raw.annotations.description]
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    codes = [extract_code(d) for d in descs]
    canon = CONFIG["CANONICAL_CODES"]
    min_dur = CONFIG["MIN_BLOCK_DURATION_S"]
    flags: list[str] = []

    candidates: list[Block] = []
    for code, label in canon.items():
        idx = sorted([i for i, c in enumerate(codes) if c == code], key=lambda i: onsets[i])
        other = np.array([onsets[i] for i, c in enumerate(codes)
                          if c in canon and c != code], dtype=float)
        for a, b in zip(idx[:-1], idx[1:]):
            t0, t1 = float(onsets[a]), float(onsets[b])
            dur = t1 - t0
            if bool(np.any((other > t0) & (other < t1))):
                reason, valid = "spans a block of the other condition", False
            elif dur < min_dur:
                reason, valid = f"duration {dur:.1f}s < {min_dur}s", False
            else:
                reason, valid = "", True
            candidates.append(Block(code, label, t0, t1, dur, valid, reason))

    selected: list[Block] = []
    for b in sorted([c for c in candidates if c.valid], key=lambda c: c.onset):
        if all(b.onset >= s.offset or b.offset <= s.onset for s in selected):
            selected.append(b)
        else:
            b.valid, b.reason = False, "overlaps an earlier selected block"

    sel = {(b.code, b.onset) for b in selected}
    for b in candidates:
        if b.valid and (b.code, b.onset) not in sel:
            b.valid, b.reason = False, "not selected"

    per_code: dict[str, int] = {}
    for b in selected:
        per_code[b.code] = per_code.get(b.code, 0) + 1
    for code, label in canon.items():
        n = per_code.get(code, 0)
        if n != CONFIG["EXPECTED_BLOCKS_PER_CODE"]:
            flags.append(f"{label} (code {code}): {n} block(s) after repair, "
                         f"expected {CONFIG['EXPECTED_BLOCKS_PER_CODE']}")

    return selected, candidates, flags


# ==============================================================================
# Export
# ==============================================================================

def export(raw, out_dir: Path, base: str, flags: list[str]) -> Path:
    import mne

    out_dir.mkdir(parents=True, exist_ok=True)

    if CONFIG["OUTPUT_FORMAT"] == "fif":
        out_path = out_dir / f"{base}_raw.fif"
        raw.save(out_path, overwrite=CONFIG["OVERWRITE"], verbose="ERROR")
        log.info(f"    wrote {out_path.name}")
        return out_path

    out_path = out_dir / f"{base}.vhdr"
    sf = raw.info["sfreq"]
    if abs(sf - round(sf)) > 1e-9:
        flags.append(f"non-integer sampling rate {sf} Hz — BrainVision stores the "
                     f"sampling interval in integer microseconds, minor rounding possible")
    try:
        mne.export.export_raw(out_path, raw, fmt="brainvision",
                              overwrite=CONFIG["OVERWRITE"], verbose="ERROR")
    except ImportError as exc:
        raise RuntimeError("BrainVision export needs pybv — pip install pybv") from exc
    log.info(f"    wrote {out_path.name} (+ .eeg, .vmrk)")
    return out_path


def copy_task_files(root: Path, out_root: Path, subject: str, dry_run: bool) -> int:
    src_dir = root / subject / CONFIG["TASK_SUBDIR"]
    if not src_dir.is_dir():
        return 0
    dst_dir = out_root / subject / CONFIG["TASK_SUBDIR"]
    n = 0
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if dry_run:
            n += 1
            continue
        dst = dst_dir / f.name
        if dst.exists() and not CONFIG["OVERWRITE"]:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    if n:
        log.info(f"  [{subject}] {'would copy' if dry_run else 'copied'} "
                 f"{n} task file(s) into the new tree")
    return n


# ==============================================================================
# Per-file driver
# ==============================================================================

def process(root: Path, out_root: Path, subject: str, rec_key: str,
            dry_run: bool) -> dict[str, Any]:
    flags: list[str] = []
    row: dict[str, Any] = {"subject": subject, "recording": rec_key,
                           "status": "pending", "timestamp_utc": utcnow()}

    paths = find_recording_files(root, subject, rec_key)
    if not paths:
        row.update(status="failed", error="no input file found", flags="missing input")
        log.error(f"  [{subject}/{rec_key}] no input file found")
        return row

    row["input_file"] = " | ".join(p.name for p in paths)
    log.info(f"  [{subject}/{rec_key}]")

    raw = load_raw(paths)
    sf = float(raw.info["sfreq"])
    row["sfreq"] = round(sf, 3)
    row["duration_s"] = round(raw.n_times / sf, 2)
    row["n_ann_total"] = len(raw.annotations)
    log.info(f"    {len(raw.ch_names)} ch, {row['duration_s']:.1f}s @ {sf:.0f} Hz, "
             f"{row['n_ann_total']} annotations")

    # ---- the repair ----
    rep = repair_annotations(raw, flags)
    row["n_renamed_1003_to_1004"] = rep["counts"].get("1003->1004", 0)
    row["n_renamed_1001_to_1002"] = rep["counts"].get("1001->1002", 0)
    row["n_renamed_total"] = rep["total"]

    codes_after = [extract_code(str(d)) for d in raw.annotations.description]
    for code in CONFIG["CANONICAL_CODES"]:
        row[f"n_ann_{code}_after"] = sum(1 for c in codes_after if c == code)

    # ---- report only ----
    valid, candidates, bflags = detect_blocks(raw)
    flags.extend(bflags)
    row["n_blocks_valid"] = len(valid)
    row["n_blocks_short"] = sum(
        1 for c in candidates
        if not c.valid and "spans a block" not in c.reason and "overlaps" not in c.reason)
    row["block_durations_s"] = " | ".join(f"{b.duration:.1f}" for b in valid)

    log.info(f"    after repair: {row['n_ann_1002_after']}x 1002, "
             f"{row['n_ann_1004_after']}x 1004  ->  {len(valid)} block(s)")
    for b in valid:
        log.info(f"      {b.label:22} {b.onset:8.1f}s -> {b.offset:8.1f}s  ({b.duration:5.1f}s)")
    if row["n_blocks_short"]:
        log.info(f"    {row['n_blocks_short']} interval(s) below "
                 f"{CONFIG['MIN_BLOCK_DURATION_S']}s (reported, not removed)")

    record: dict[str, Any] = {
        "subject": subject, "recording": rec_key, "timestamp_utc": utcnow(),
        "input_files": [str(p) for p in paths],
        "sfreq": sf, "duration_s": row["duration_s"],
        "n_annotations": row["n_ann_total"],
        "repair": rep,
        "blocks_after_repair": [
            {"code": c.code, "label": c.label, "onset_s": round(c.onset, 3),
             "offset_s": round(c.offset, 3), "duration_s": round(c.duration, 3),
             "valid": c.valid, "reason": c.reason}
            for c in sorted(candidates, key=lambda x: x.onset)
        ],
        "config": CONFIG,
    }

    out_dir = out_root / subject
    if CONFIG["OUTPUT_SUBDIR"]:
        out_dir = out_dir / CONFIG["OUTPUT_SUBDIR"]
    base = f"{subject}_{rec_key}{CONFIG['OUTPUT_SUFFIX']}"

    if dry_run:
        ext = "_raw.fif" if CONFIG["OUTPUT_FORMAT"] == "fif" else ".vhdr"
        row["output_file"] = str(out_dir / f"{base}{ext}")
        log.info(f"    DRY RUN — would write {row['output_file']}")
        row["status"] = "dry_run"
        row["flags"] = "; ".join(flags)
        return row

    out_path = export(raw, out_dir, base, flags)
    row["output_file"] = str(out_path)
    record["output_file"] = str(out_path)

    (out_dir / f"{base}_repair_log.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")

    row["status"] = "done"
    row["flags"] = "; ".join(flags)
    return row


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repair mis-typed rest-state annotation markers (1003->1004, 1001->1002)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output-root", type=Path, default=None,
                    help=f"root of the new tree (default: <root>/{CONFIG['OUTPUT_ROOT_NAME']})")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--recordings", nargs="*", default=None,
                    choices=list(CONFIG["REST_RECORDINGS"]))
    ap.add_argument("--format", choices=["brainvision", "fif"], default=None,
                    help="output format (default brainvision; fif is lossless)")
    ap.add_argument("--copy-task", action="store_true",
                    help="also copy the untouched task recordings into the new tree")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
        return 2

    out_root = (args.output_root or root / CONFIG["OUTPUT_ROOT_NAME"]).expanduser().resolve()
    if out_root == root:
        print("ERROR: --output-root must not be the same as --root; repaired files "
              "are always kept separate from the raw data.", file=sys.stderr)
        return 2
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    if args.format:
        CONFIG["OUTPUT_FORMAT"] = args.format
    if args.copy_task:
        CONFIG["COPY_TASK_FILES"] = True

    setup_logging(out_root / "derivatives" / "repair_annotations_run.log",
                  verbose=not args.quiet)

    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        log.error("MNE is not installed. pip install 'mne>=1.6' pybv")
        return 2

    ext = "_raw.fif" if CONFIG["OUTPUT_FORMAT"] == "fif" else ".vhdr"
    log.info("=" * 78)
    log.info("ANNOTATION MARKER REPAIR" + ("  [DRY RUN]" if args.dry_run else ""))
    log.info(f"source root  : {root}")
    log.info(f"output root  : {out_root}   (NEW tree — raw data untouched)")
    log.info(f"output layout: {out_root.name}/hx_XX/"
             + (f"{CONFIG['OUTPUT_SUBDIR']}/" if CONFIG["OUTPUT_SUBDIR"] else "")
             + f"hx_XX_<rec>{CONFIG['OUTPUT_SUFFIX']}{ext}")
    log.info("repair       : " + ", ".join(f"{a} -> {c}"
                                           for a, c in CONFIG["CODE_ALIASES"].items()))
    log.info("signal       : NOT modified — annotations only")
    if CONFIG["COPY_TASK_FILES"]:
        log.info("task files   : copied into the new tree (--copy-task)")
    log.info("=" * 78)

    subjects = args.subjects or find_subjects(root)
    rec_keys = args.recordings or list(CONFIG["REST_RECORDINGS"])
    rows: list[dict[str, Any]] = []

    for sub in subjects:
        if sub in CONFIG["EXCLUDE_SUBJECTS"]:
            log.info(f"[{sub}] skipped — excluded_by_request")
            rows.append({"subject": sub, "recording": "*", "status": "excluded_by_request",
                         "timestamp_utc": utcnow()})
            continue
        for rec in rec_keys:
            try:
                rows.append(process(root, out_root, sub, rec, args.dry_run))
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error(f"  [{sub}/{rec}] FAILED: {exc}")
                log.debug(tb)
                rows.append({"subject": sub, "recording": rec, "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}",
                             "timestamp_utc": utcnow()})
        if CONFIG["COPY_TASK_FILES"]:
            copy_task_files(root, out_root, sub, args.dry_run)

    # Stage 0 looks for the montage at its own root — carry it across so the
    # new tree is usable on its own.
    if not args.dry_run:
        for elc in sorted(root.glob("*.elc")):
            dst = out_root / elc.name
            if not dst.exists() or CONFIG["OVERWRITE"]:
                shutil.copy2(elc, dst)
                log.info(f"copied montage {elc.name} into the new tree")

    df = pd.DataFrame(rows)
    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[MANIFEST_COLUMNS].astype(object)
    man = out_root / "derivatives" / (
        "repair_annotations_manifest_dryrun.csv" if args.dry_run
        else "repair_annotations_manifest.csv")
    man.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(man, index=False)

    counts = df["status"].value_counts().to_dict()
    total_fixed = pd.to_numeric(df["n_renamed_total"], errors="coerce").sum()
    log.info("=" * 78)
    log.info(" · ".join(f"{k}: {v}" for k, v in counts.items()))
    log.info(f"markers repaired across all files: {int(total_fixed)}")
    log.info(f"manifest : {man}")
    log.info("=" * 78)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
