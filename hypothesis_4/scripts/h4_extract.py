# -*- coding: utf-8 -*-
"""
Hypothesis 4, pass 1 -- beta power during gait, one row per gait epoch.

Each gait epoch runs from the walking-onset marker to the walking-end marker.
Epochs outside the duration bounds are dropped as interrupted trials. The rest
are cut into SEGMENT_S segments, which serve two purposes:

    AutoReject needs equal-length epochs, which variable-length walks cannot
    provide, so it judges the segments.

    Welch's resolution is sfreq / n_fft, so an epoch-length window would give
    every walk a different frequency axis. With n_fft fixed at the segment
    length every trial lands on the same grid.

The clean segments of a trial are then collapsed into ONE spectrum for that
trial, specparam removes the aperiodic component, and beta power is the area
under the flattened spectrum. So each walk still yields exactly one value.

The z > 2 fit screen is NOT applied here: the protocol z-scores the fit metric
across all participants, so that waits for h4_group.py.

    python h4_extract.py
    python h4_group.py

Writes into OUTPUT_DIR:
    h4_epoch_level.csv   one row per surviving gait epoch
    h4_run.log
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

import h4_config as cfg
from h4_trials import find_trials
from h4_spectra import (find_participants, load_participant_info,
                        roi_for_participant, segment_starts, segment_events,
                        segment_spectra, flattened_beta_auc, average, output_tag)

mne.set_log_level("ERROR")

log = logging.getLogger("h4")


def setup_logging():
    logfile = os.path.join(cfg.OUTPUT_DIR, "h4_run.log")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    for handler in (logging.FileHandler(logfile, mode="a", encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


def split_by_duration(trials):
    """(kept, n_short, n_long). Interrupted trials are those whose walk was too
    brief or too long to be a normal completed traverse of the path."""
    kept, n_short, n_long = [], 0, 0
    for trial in trials:
        if trial["duration_s"] < cfg.MIN_DURATION_S:
            n_short += 1
        elif trial["duration_s"] > cfg.MAX_DURATION_S:
            n_long += 1
        else:
            kept.append(trial)
    return kept, n_short, n_long


def clean_segments(raw, trials, roi_channels):
    """Segment every trial, let AutoReject judge the segments, and return the
    clean ones.

    Returns (freqs, psd, owner, owner_all):
        psd        (n_clean_segments, n_freqs), one row per surviving segment
        owner      the trial index of each CLEAN segment
        owner_all  the trial index of every segment MNE cut, clean or not, so
                   the flagged fraction of a trial can be worked out

    All spectra share one frequency axis because every segment is the same
    length -- which is the whole reason for segmenting.
    """
    sfreq = raw.info["sfreq"]
    events, owner = segment_events(trials, sfreq)
    if len(events) == 0:
        return None, None, None

    # tmax is one sample short of SEGMENT_S so each epoch holds exactly the
    # segment length; MNE's tmax is inclusive.
    segment_epochs = mne.Epochs(raw, events, event_id=cfg.CONDITION_IDS,
                                tmin=0.0, tmax=cfg.SEGMENT_S - 1.0 / sfreq,
                                baseline=None, picks=roi_channels,
                                preload=True, verbose="ERROR")
    if len(segment_epochs) == 0:
        return None, None, None

    autoreject = AutoReject(n_interpolate=cfg.AR_N_INTERPOLATE,
                            consensus=cfg.AR_CONSENSUS, cv=cfg.AR_CV,
                            random_state=cfg.RANDOM_STATE, n_jobs=1,
                            verbose=False)
    autoreject.fit(segment_epochs)
    bad = autoreject.get_reject_log(segment_epochs).bad_epochs

    # segment_epochs.selection indexes into `events`, so it maps back to owner.
    owner_present = owner[segment_epochs.selection]

    freqs, psd = segment_spectra(segment_epochs)
    return freqs, psd[~bad], owner_present[~bad], owner_present


def process_participant(participant, path, roi_channels, hemisphere):
    """Every gait epoch for one participant."""
    result = {"participant": participant, "excluded": None, "rows": [],
              "spectra": [], "flattened": [], "epoch_ids": [], "freqs": None,
              "message": ""}

    try:
        raw = mne.io.read_raw_fif(path, preload=False)
        trials = [t for t in find_trials(raw) if t["status"] == "ok"]
        if not trials:
            result["excluded"] = "no valid gait epochs"
            return result

        trials, n_short, n_long = split_by_duration(trials)
        if not trials:
            result["excluded"] = ("every gait epoch was interrupted (%d short, "
                                  "%d long)" % (n_short, n_long))
            return result

        before = {c: sum(t["condition"] == c for t in trials)
                  for c in cfg.CONDITION_IDS}

        segmented = clean_segments(raw, trials, roi_channels)
        if segmented[0] is None:
            result["excluded"] = "no segments could be cut"
            return result
        freqs, psd, owner, owner_all = segmented

        n_segments_total = len(owner_all)
        n_segments_clean = len(owner)

        n_failed = 0
        for index, trial in enumerate(trials):
            mine = psd[owner == index]
            total_mine = int(np.sum(owner_all == index))
            if total_mine == 0:
                n_failed += 1
                continue

            flagged_fraction = 1.0 - (len(mine) / total_mine)
            if (len(mine) < cfg.AR_MIN_CLEAN_SEGMENTS
                    or flagged_fraction > cfg.AR_MAX_FLAGGED_FRACTION):
                continue

            # The clean segments of this walk become one spectrum for the walk.
            trial_psd = average(mine, axis=0)

            auc, fit, model_freqs, original, flattened = flattened_beta_auc(
                freqs, trial_psd)
            if auc is None:
                n_failed += 1
                continue

            # epoch_id ties this row to its spectra, so h4_group.py can plot
            # exactly the epochs that survive the fit screen.
            epoch_id = "%s_%d" % (participant, index)
            result["rows"].append({
                "epoch_id": epoch_id,
                "participant": participant,
                "condition": trial["condition"],
                "direction": trial["direction"],
                "hemisphere": hemisphere,
                "duration_s": round(trial["duration_s"], 3),
                "n_segments": len(mine),
                "beta_auc": auc,
                "fit": fit,
            })
            result["spectra"].append(original)
            result["flattened"].append(flattened)
            result["epoch_ids"].append(epoch_id)
            result["freqs"] = model_freqs

        after = {c: sum(r["condition"] == c for r in result["rows"])
                 for c in cfg.CONDITION_IDS}
        if min(after.values()) < cfg.MIN_TRIALS:
            result["excluded"] = ("%d ON and %d OFF gait epochs after AutoReject"
                                  % (after["ON"], after["OFF"]))
            result["rows"] = []
            result["spectra"], result["flattened"], result["epoch_ids"] = [], [], []
            return result

        result["message"] = (
            "%-8s ON %d/%d   OFF %d/%d   interrupted %d (%d short, %d long)   "
            "segments %d/%d clean"
            % (participant, after["ON"], before["ON"], after["OFF"], before["OFF"],
               n_short + n_long, n_short, n_long,
               n_segments_clean, n_segments_total))
        if n_failed:
            result["message"] += "   specparam failed on %d" % n_failed
    except Exception as error:
        result["excluded"] = "%s: %s" % (type(error).__name__, error)

    return result


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logging()

    participants = find_participants()
    participant_info = load_participant_info()

    jobs = []
    for participant, path in participants:
        roi_channels, hemisphere = roi_for_participant(participant_info, participant)
        jobs.append((participant, path, roi_channels, hemisphere))

    results = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(process_participant)(*job) for job in jobs)

    rows, spectra, flattened, epoch_ids, freqs = [], [], [], [], None
    for result in results:
        if result["excluded"]:
            log.info("%-8s excluded: %s", result["participant"], result["excluded"])
            continue
        log.info(result["message"])
        rows.extend(result["rows"])
        spectra.extend(result["spectra"])
        flattened.extend(result["flattened"])
        epoch_ids.extend(result["epoch_ids"])
        if result["freqs"] is not None:
            freqs = result["freqs"]

    if not rows:
        log.info("No gait epochs survived.")
        return

    tag = output_tag()
    if tag:
        log.info("AVERAGING = '%s' -- outputs tagged '%s'", cfg.AVERAGING, tag)

    epoch_csv = os.path.join(cfg.OUTPUT_DIR, "h4_epoch_level%s.csv" % tag)
    pd.DataFrame(rows).to_csv(epoch_csv, index=False)

    # Every epoch's spectra, keyed by epoch_id, so h4_group.py can plot exactly
    # the ones that survive the fit screen without recomputing anything.
    np.savez_compressed(
        os.path.join(cfg.OUTPUT_DIR, "h4_epoch_spectra%s.npz" % tag),
        epoch_ids=np.array(epoch_ids), freqs=freqs,
        original=np.stack(spectra), flattened=np.stack(flattened))

    log.info("%d gait epochs -> %s", len(rows), epoch_csv)
    log.info("Now run h4_group.py to apply the z > %.1f fit screen.",
             cfg.FIT_Z_THRESHOLD)


if __name__ == "__main__":
    main()
