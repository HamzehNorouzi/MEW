# -*- coding: utf-8 -*-
"""
Shared ERS computation for the Hypothesis 3 pipeline.

The one thing to understand here: a trial's baseline power and its task power
come from two epochs cut at two different markers. The baseline epoch is locked
to MOVEMENT ONSET (so the -3 to -2 s standing window the protocol asks for is
available); the analysis epoch is locked to MOVEMENT STOP. Both TFRs use
identical wavelets, so their power values are directly comparable, and the
within-trial normalisation divides one by the other.
"""

import os
import glob

import numpy as np
import pandas as pd
import mne

import h3_config as cfg
from h3_trials import make_events

# The moving average is centred, so a point needs half a window of data on each
# side. The analysis TFR is kept this much wider than TMIN-TMAX and trimmed back
# after smoothing, so real data fills the kernel instead of zero-padding pulling
# the ERS toward zero at the window edges. Sized to the wider of the two
# smoothing windows in use.
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
    """Suffix for filenames, combining the averaging and baseline settings, so
    an exploratory run cannot overwrite a protocol one."""
    tag = "" if cfg.AVERAGING == "median" else "_mean"
    if cfg.BASELINE_MODE != "onset":
        tag += "_baseline-stop"
    return tag


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


def build_epochs(raw, accepted_trials, picks=None):
    """The two trial-matched epoch sets for a set of accepted trials.

    Returns (baseline_epochs, task_epochs, kept), where `kept` indexes into
    accepted_trials. A trial too close to either end of the recording is dropped
    from whichever set cannot hold it; only trials surviving in BOTH sets are
    kept, so the two stay trial-for-trial aligned.
    """
    baseline_events, task_events = make_events(accepted_trials)
    if len(task_events) == 0:
        return None, None, np.array([], dtype=int)

    baseline_epochs = mne.Epochs(raw, baseline_events,
                                 tmin=cfg.BASE_TMIN_PAD, tmax=cfg.BASE_TMAX_PAD,
                                 baseline=None, picks=picks, preload=True,
                                 verbose="ERROR")
    task_epochs = mne.Epochs(raw, task_events,
                             tmin=cfg.TMIN_PAD, tmax=cfg.TMAX_PAD,
                             baseline=None, picks=picks, preload=True,
                             verbose="ERROR")

    kept = np.intersect1d(baseline_epochs.selection, task_epochs.selection)
    if len(kept) == 0:
        return None, None, kept

    baseline_epochs = baseline_epochs[np.where(np.isin(baseline_epochs.selection, kept))[0]]
    task_epochs = task_epochs[np.where(np.isin(task_epochs.selection, kept))[0]]
    return baseline_epochs, task_epochs, kept


def _tfr(epochs):
    return epochs.compute_tfr(method="morlet", freqs=cfg.FREQS, n_cycles=cfg.N_CYCLES,
                              decim=cfg.DECIM, average=False, return_itc=False,
                              verbose="ERROR")


def percent_change(baseline_epochs, task_epochs):
    """Per-trial beta power after movement stop, as percent change from that
    trial's own pre-onset standing baseline.

    Both epoch sets must already be picked down to the ROI channels and be
    trial-matched. Returns:
        percent  (n_epochs, n_channels, n_freqs, n_times)
        times    time from movement stop, still carrying the smoothing margin
    """
    task_tfr = _tfr(task_epochs)

    if cfg.BASELINE_MODE == "stop":
        # Exploratory: the baseline is the second before movement stop, taken
        # from the analysis epoch itself. Read it BEFORE the crop below, while
        # the pre-stop samples are still present.
        window = ((task_tfr.times >= cfg.BASELINE_STOP[0] - 1e-6)
                  & (task_tfr.times <= cfg.BASELINE_STOP[1] + 1e-6))
        baseline = average(task_tfr.data[..., window], axis=-1, keepdims=True)
    else:
        # The protocol: quiet standing before gait, from the onset-locked epoch.
        baseline_tfr = _tfr(baseline_epochs)
        baseline_tfr.crop(cfg.BASELINE[0], cfg.BASELINE[1])
        baseline = average(baseline_tfr.data, axis=-1, keepdims=True)

    task_tfr.crop(cfg.TMIN - SMOOTH_PAD_S, cfg.TMAX + SMOOTH_PAD_S)

    percent = 100.0 * (task_tfr.data - baseline) / baseline
    return percent, task_tfr.times


def condition_percent_change(raw, condition_trials, roi_channels):
    """percent_change() straight from a raw file, for one condition.

    Used by the inspection script, which has no AutoReject step and so can build
    one condition at a time. Returns (None, None) if there are no trials.
    """
    if not condition_trials:
        return None, None

    baseline_epochs, task_epochs, kept = build_epochs(raw, condition_trials,
                                                      picks=roi_channels)
    if baseline_epochs is None:
        return None, None
    return percent_change(baseline_epochs, task_epochs)


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

    The shared starting point for both the TFR image and the ERS time course,
    so a noisy trial cannot dominate either one.
    """
    return average(percent, axis=0)


def to_tfr_image(by_trial, times, smooth_ms):
    """(channels, freqs, times) -> smoothed, trimmed (freqs, times), for the
    group TFR plot. Frequency is kept, only channels are collapsed."""
    tfr_2d = average(by_trial, axis=0)               # (freqs, times)
    tfr_2d = smooth(tfr_2d, times, smooth_ms)
    return tfr_2d[:, _trim_mask(times)]


def to_timecourse(by_trial, times, smooth_ms):
    """(channels, freqs, times) -> smoothed, trimmed (times,): the 1-D ERS
    curve. Frequency is collapsed before channels."""
    by_freq = average(by_trial, axis=1)              # (channels, times)
    timecourse = average(by_freq, axis=0)            # (times,)
    timecourse = smooth(timecourse, times, smooth_ms)
    return timecourse[_trim_mask(times)]
