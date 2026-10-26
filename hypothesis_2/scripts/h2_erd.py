# -*- coding: utf-8 -*-
"""
Shared ERD computation for the Hypothesis 2 pipeline.

Used by both h2_inspect.py (a quick, AutoReject-free look at the ERD, to check
the epoching before committing to a full run) and h2_extract.py (the full
pipeline). Keeping the computation in one place means the two can never quietly
disagree about what an ERD is.
"""

import os
import glob

import numpy as np
import pandas as pd
import mne

import h2_config as cfg
from h2_trials import make_events

# The moving average is centred, so a point needs half a window of data on each
# side. The TFR is therefore kept this much wider than the analysis window and
# trimmed back after smoothing, so real data fills the kernel instead of
# zero-padding pulling the ERD toward zero exactly at t = 0. Sized to the wider
# of the two smoothing windows in use (SMOOTH_MS, PLOT_SMOOTH_MS) so both are
# safe from the same padded crop.
SMOOTH_PAD_S = max(cfg.SMOOTH_MS, cfg.PLOT_SMOOTH_MS) / 2000.0


def average(x, axis=None, keepdims=False):
    """Collapse an array using whichever of median/mean cfg.AVERAGING selects.

    Every scientific averaging step in this pipeline goes through here, so the
    two settings cannot drift apart. Note that np.median and np.mean take
    identical arguments, so nothing else has to change when the setting does.

    This deliberately does NOT cover descriptive summaries elsewhere (walk
    durations in the diagnostics, or the sampling rate derived from the time
    vector) -- those are not averaging the signal.
    """
    if cfg.AVERAGING not in ("median", "mean"):
        raise ValueError("AVERAGING must be 'median' or 'mean', got %r"
                         % cfg.AVERAGING)
    function = np.mean if cfg.AVERAGING == "mean" else np.median
    return function(x, axis=axis, keepdims=keepdims)


def output_tag():
    """Suffix for filenames, so a mean run cannot overwrite a median one."""
    return "" if cfg.AVERAGING == "median" else "_mean"


def find_participants():
    """(participant_id, path) for every task file under DERIVATIVES_DIR."""
    pattern = os.path.join(cfg.DERIVATIVES_DIR, "*", "*_task_clean-raw.fif")
    return [(os.path.basename(os.path.dirname(path)), path)
            for path in sorted(glob.glob(pattern))]


def load_participant_info():
    return pd.read_csv(cfg.PARTICIPANT_INFO)


def roi_for_participant(participant_info, participant_id):
    """ROI channel names for one participant: the hemisphere opposite the
    stimulator."""
    row = participant_info.loc[participant_info[cfg.ID_COLUMN] == participant_id]
    if row.empty:
        raise KeyError("%s not found in %s" % (participant_id, cfg.PARTICIPANT_INFO))

    side = str(row.iloc[0][cfg.SIDE_COLUMN]).strip().lower()
    if side.startswith("r"):
        return cfg.ROI_LEFT
    if side.startswith("l"):
        return cfg.ROI_RIGHT
    raise ValueError("%s: unrecognised %s value %r"
                     % (participant_id, cfg.SIDE_COLUMN, row.iloc[0][cfg.SIDE_COLUMN]))


def percent_change(epochs):
    """Per-trial beta power as percent change from that trial's own baseline.

    `epochs` must already be picked down to the ROI channels and cover the
    padded window. Returns:
        percent  (n_epochs, n_channels, n_freqs, n_times)
        times    the time vector, still carrying the smoothing margin
    """
    tfr = epochs.compute_tfr(method="morlet", freqs=cfg.FREQS, n_cycles=cfg.N_CYCLES,
                             decim=cfg.DECIM, average=False, return_itc=False,
                             verbose="ERROR")
    tfr.crop(cfg.TMIN - SMOOTH_PAD_S, cfg.TMAX + SMOOTH_PAD_S)

    baseline_mask = ((tfr.times >= cfg.BASELINE[0] - 1e-6)
                     & (tfr.times <= cfg.BASELINE[1] + 1e-6))
    baseline = average(tfr.data[..., baseline_mask], axis=-1, keepdims=True)
    percent = 100.0 * (tfr.data - baseline) / baseline
    return percent, tfr.times


def condition_percent_change(raw, condition_trials, roi_channels):
    """percent_change() straight from a raw file, for one condition.

    Used by the inspection script, which has no AutoReject step and so can build
    one condition at a time. Returns (None, None) if there are no trials.
    """
    if not condition_trials:
        return None, None

    events = make_events(condition_trials)
    epochs = mne.Epochs(raw, events, tmin=cfg.TMIN_PAD, tmax=cfg.TMAX_PAD,
                        baseline=None, picks=roi_channels, preload=True,
                        verbose="ERROR")
    if len(epochs) == 0:
        return None, None
    return percent_change(epochs)


def smooth(x, times, window_ms):
    """Centred moving average along the last axis. Output is the same length."""
    sfreq = 1.0 / np.median(np.diff(times))
    n = max(1, int(round(window_ms / 1000.0 * sfreq)))
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"),
                               axis=-1, arr=x)


def _trim_mask(times):
    return (times >= cfg.TMIN - 1e-6) & (times <= cfg.TMAX + 1e-6)


def trimmed_times(times):
    """times, cut back from the padded window to TMIN-TMAX."""
    return times[_trim_mask(times)]


def collapse_trials(percent):
    """(n_epochs, n_channels, n_freqs, n_times) percent-change array -> median
    across trials only: (n_channels, n_freqs, n_times).

    The shared starting point for both the TFR image and the ERD time course,
    so a noisy trial cannot dominate either one, and so the two never
    accidentally collapse trials differently.
    """
    return average(percent, axis=0)


def to_tfr_image(by_trial, times, smooth_ms):
    """(channels, freqs, times) -> smoothed, trimmed (freqs, times), for the
    group TFR plot. Frequency is kept, only channels are collapsed."""
    tfr_2d = average(by_trial, axis=0)               # (freqs, times)
    tfr_2d = smooth(tfr_2d, times, smooth_ms)
    return tfr_2d[:, _trim_mask(times)]


def to_timecourse(by_trial, times, smooth_ms):
    """(channels, freqs, times) -> smoothed, trimmed (times,): the 1-D ERD
    curve. Frequency is collapsed before channels."""
    by_freq = average(by_trial, axis=1)              # (channels, times)
    timecourse = average(by_freq, axis=0)            # (times,)
    timecourse = smooth(timecourse, times, smooth_ms)
    return timecourse[_trim_mask(times)]
