# -*- coding: utf-8 -*-
"""
Group-level spectra figures for the Hypothesis 4 pipeline.

Two panels on one figure:

    left    the original Welch spectra, in log10 power, that specparam was
            fitted to. The 1/f slope dominates here, which is exactly why the
            aperiodic component has to be removed before beta is quantified.

    right   the flattened spectra, i.e. the same data with each trial's own
            aperiodic fit subtracted. This is what the beta AUC integrates, so
            the shaded 14-35 Hz band is the quantity H4 compares.

Both panels show the grand median (or mean, per AVERAGING) across participants,
with each participant's own spectrum faint behind it.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h4_config as cfg

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _panel(ax, freqs, by_condition, collapse, title, ylabel, show_band_label):
    for condition, colour in (("ON", cfg.COLOR_ON), ("OFF", cfg.COLOR_OFF)):
        traces = by_condition.get(condition)
        if not traces:
            continue
        stacked = np.stack(traces)
        for trace in stacked:
            ax.plot(freqs, trace, color=colour, lw=0.6, alpha=0.18)
        ax.plot(freqs, collapse(stacked, axis=0), color=colour, lw=2.4,
                label="Stim %s" % condition)

    ax.axvspan(cfg.BETA_BAND[0], cfg.BETA_BAND[1], color=GRID, alpha=0.55,
               zorder=0, label="beta 14-35 Hz" if show_band_label else None)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(cfg.SPECPARAM_FREQ_RANGE)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _style_axis(ax)


def plot_group_spectra(freqs, original_by_condition, flattened_by_condition,
                       collapse, out_path):
    """Original and flattened spectra side by side.

    `collapse` is the averaging function (median or mean, per AVERAGING), passed
    in so this module never has to decide how the data is collapsed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor=SURFACE,
                             constrained_layout=True)

    _panel(axes[0], freqs, original_by_condition, collapse,
           "Gait spectra (specparam input)", "log10 power", False)
    _panel(axes[1], freqs, flattened_by_condition, collapse,
           "Flattened (aperiodic removed)", "log10 power, aperiodic-adjusted",
           True)
    axes[1].axhline(0, color=GRID, lw=1, zorder=1)

    fig.savefig(out_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
