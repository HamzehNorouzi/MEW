# -*- coding: utf-8 -*-
"""
Shared spectral computation for the Hypothesis 4 pipeline.

One gait epoch -> one Welch spectrum per ROI -> specparam -> the area under the
flattened spectrum between 14 and 35 Hz.

Each epoch is independent and is used at its own natural duration. Welch runs
over the whole epoch as a single window, so nothing is segmented and epochs of
different lengths are each described by one spectrum. Because H4 asks about
sustained power rather than about change over time, the differing durations do
not need reconciling.
"""

import os
import glob

import numpy as np
import pandas as pd
import mne

import h4_config as cfg

# specparam v1 renamed the package; FOOOF is the older name for the same model.
try:
    from specparam import SpectralModel as _SpectralModel
    _SPECPARAM_V1 = True
except ImportError:                                    # pragma: no cover
    from fooof import FOOOF as _SpectralModel
    _SPECPARAM_V1 = False


def average(x, axis=None, keepdims=False):
    """Collapse an array using whichever of median/mean cfg.AVERAGING selects."""
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
    """(contralateral_channels, hemisphere_name) for one participant."""
    row = participant_info.loc[participant_info[cfg.ID_COLUMN] == participant_id]
    if row.empty:
        raise KeyError("%s not found in %s" % (participant_id, cfg.PARTICIPANT_INFO))

    side = str(row.iloc[0][cfg.SIDE_COLUMN]).strip().lower()
    if side.startswith("r"):
        return cfg.ROI_LEFT, "left"
    if side.startswith("l"):
        return cfg.ROI_RIGHT, "right"
    raise ValueError("%s: unrecognised %s value %r"
                     % (participant_id, cfg.SIDE_COLUMN, row.iloc[0][cfg.SIDE_COLUMN]))


def segment_starts(trial, sfreq):
    """Sample offsets of the consecutive segments tiling one gait epoch.

    Offsets are relative to the trial's onset sample. A remainder shorter than
    one full segment is not used, so every segment is exactly SEGMENT_S long.
    """
    step = int(round(cfg.SEGMENT_S * sfreq))
    hop = max(1, int(round(step * (1.0 - cfg.SEGMENT_OVERLAP))))
    span = trial["end_sample"] - trial["onset_sample"]
    return list(range(0, span - step + 1, hop)), step


def segment_events(trials, sfreq):
    """One event per segment across all trials, plus the index of the trial each
    segment belongs to. This is what AutoReject is fitted on."""
    rows, owner = [], []
    for index, trial in enumerate(trials):
        starts, _ = segment_starts(trial, sfreq)
        for start in starts:
            rows.append([trial["onset_sample"] + start, 0,
                         cfg.CONDITION_IDS[trial["condition"]]])
            owner.append(index)
    return (np.array(rows, dtype=int).reshape(-1, 3),
            np.array(owner, dtype=int))


def segment_spectra(segment_epochs):
    """Welch PSD for every segment, spatially averaged over the ROI.

    n_fft is the segment length, which is identical for every segment of every
    trial, so all spectra share one frequency axis. Returns (freqs, psd) with
    psd shaped (n_segments, n_freqs).
    """
    n_fft = segment_epochs.get_data().shape[-1]
    spectrum = segment_epochs.compute_psd(method="welch", fmin=cfg.PSD_FMIN,
                                          fmax=cfg.PSD_FMAX, n_fft=n_fft,
                                          n_per_seg=n_fft, n_overlap=0,
                                          verbose="ERROR")
    psd = spectrum.get_data()                       # (segments, channels, freqs)
    return spectrum.freqs, average(psd, axis=1)     # collapse the ROI channels


# ---- specparam version shim ---------------------------------------------------
# specparam 2.0 moved everything off the model itself: the input spectrum lives
# on `model.data`, the fitted results (params, has_model, get_params, ...) live
# on `model.results`, and the fitted curves (_ap_fit, get_component, ...) live
# one level deeper on `model.results.model`. Older specparam (<2.0) and the
# original `fooof.FOOOF` keep all of this as flat attributes directly on the
# model (`model.freqs`, `model.aperiodic_params_`, `model._ap_fit`, ...).
#
# These helpers try the new nested layout first and fall back to the old flat
# one, so the rest of the module does not need to know which version is
# installed.

def _results_of(model):
    return getattr(model, "results", model)


def _data_of(model):
    return getattr(model, "data", model)


def _fit_curves_of(model):
    results = _results_of(model)
    return getattr(results, "model", results)


def _get_freqs(model):
    return np.asarray(getattr(_data_of(model), "freqs"))


def _get_power_spectrum(model):
    return np.asarray(getattr(_data_of(model), "power_spectrum"))


def _has_model(model):
    value = getattr(_results_of(model), "has_model", None)
    if value is not None:
        return bool(value)
    return getattr(model, "aperiodic_params_", None) is not None


def _get_aperiodic_params(model):
    results = _results_of(model)
    if hasattr(results, "get_params"):
        return np.atleast_1d(results.get_params("aperiodic"))
    if hasattr(model, "get_params"):
        return np.atleast_1d(model.get_params("aperiodic"))
    return np.atleast_1d(model.aperiodic_params_)


def _fit_metric(model):
    """specparam's goodness-of-fit metric, across package versions."""
    name = cfg.SPECPARAM_FIT_METRIC
    value = getattr(model, name + "_", None)
    if value is not None:
        return float(value)
    for container in (model, _results_of(model)):
        if hasattr(container, "get_metrics"):
            try:
                value = container.get_metrics(name)
            except Exception:
                continue
            if value is not None and np.size(value) > 0:
                return float(np.asarray(value).ravel()[0])
    raise AttributeError("could not read the %r fit metric from this specparam "
                         "model -- specparam's API has changed again" % name)


def _aperiodic_fit(model):
    """The fitted aperiodic component, in log10 power.

    specparam exposes this under different names and in different places across
    versions, and none of them are guaranteed, so try each in turn rather than
    let one rename break the whole pipeline silently.
    """
    for container in (_fit_curves_of(model), _results_of(model), model):
        for name in ("_ap_fit", "ap_fit_", "_aperiodic_fit"):
            value = getattr(container, name, None)
            if value is not None:
                return np.asarray(value)

    for container in (_fit_curves_of(model), _results_of(model), model):
        if hasattr(container, "get_component"):
            try:
                return np.asarray(container.get_component("aperiodic", "log"))
            except Exception:
                pass

    # Last resort: rebuild it from the fitted parameters.
    try:
        from specparam.sim.gen import gen_aperiodic
    except ImportError:
        try:
            from fooof.sim.gen import gen_aperiodic
        except ImportError:
            return None
    return np.asarray(gen_aperiodic(_get_freqs(model), _get_aperiodic_params(model)))


def flattened_beta_auc(freqs, psd):
    """Fit specparam, subtract the aperiodic component, and integrate the
    flattened spectrum over the beta band.

    Returns (auc, fit_metric, model_freqs, original, flattened), or five Nones
    if the model does not converge. The spectra are returned so the plots can
    reuse them rather than refitting.

    The flattened spectrum is in log10 power, which is specparam's own scale and
    the scale on which the aperiodic component is defined; subtracting on any
    other scale would not remove it.
    """
    nothing = (None, None, None, None, None)

    model = _SpectralModel(
        peak_width_limits=cfg.SPECPARAM_PEAK_WIDTH_LIMITS,
        max_n_peaks=cfg.SPECPARAM_MAX_N_PEAKS,
        min_peak_height=cfg.SPECPARAM_MIN_PEAK_HEIGHT,
        peak_threshold=cfg.SPECPARAM_PEAK_THRESHOLD,
        aperiodic_mode=cfg.SPECPARAM_APERIODIC_MODE,
        verbose=False,
    )
    try:
        model.fit(freqs, psd, cfg.SPECPARAM_FREQ_RANGE)
    except Exception:
        return nothing

    if not _has_model(model) or np.any(np.isnan(_get_aperiodic_params(model))):
        return nothing

    aperiodic = _aperiodic_fit(model)
    power_spectrum = _get_power_spectrum(model)
    if aperiodic is None or aperiodic.shape != power_spectrum.shape:
        return nothing

    original = power_spectrum                             # log10 power
    flattened = original - aperiodic                      # log10 power

    model_freqs = _get_freqs(model)
    band = ((model_freqs >= cfg.BETA_BAND[0] - 1e-9)
            & (model_freqs <= cfg.BETA_BAND[1] + 1e-9))
    if not np.any(band):
        return nothing

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapezoid(flattened[band], model_freqs[band]))
    return auc, _fit_metric(model), model_freqs, original, flattened
