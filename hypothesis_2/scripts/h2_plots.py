# -*- coding: utf-8 -*-
"""
Group-level ERD figures, shared by h2_inspect.py and h2_extract.py.

Both figures use the sign convention required throughout this pipeline: ERD is
negative, so blue is negative in the TFR and a deeper valley is a larger ERD in
the time course.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import h2_config as cfg

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Diverging blue <-> red, gray at zero. Blue = ERD (negative), red = ERS (positive).
ERD_CMAP = LinearSegmentedColormap.from_list("erd", ["#2a78d6", "#f0efec", "#e34948"])


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_group_tfr(times, freqs, tfr_on, tfr_off, out_path):
    """Two-panel beta TFR, grand median across participants, ON and OFF.

    Colour range is clipped to a percentile of |ERD %| rather than the literal
    max, so a few outlier pixels don't stretch the scale and flatten the
    contrast in the range that actually carries the ERD. Values beyond the
    clip still show, fully saturated at the colormap's end.
    """
    combined = np.concatenate([np.abs(tfr_on).ravel(), np.abs(tfr_off).ravel()])
    vmax = np.percentile(combined, cfg.TFR_COLOR_PERCENTILE)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True,
                             constrained_layout=True, facecolor=SURFACE)
    for ax, data, label in zip(axes, [tfr_on, tfr_off], ["Stim ON", "Stim OFF"]):
        im = ax.pcolormesh(times, freqs, data, cmap=ERD_CMAP, norm=norm,
                           shading="gouraud")
        ax.axvline(0, color=INK_SOFT, lw=1, ls="--")
        ax.set_title(label, color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time from movement onset (s)")
        _style_axis(ax)
    axes[0].set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(im, ax=axes, label="ERD (%)", shrink=0.9)
    cbar.ax.tick_params(colors=INK_SOFT, labelsize=9)
    cbar.set_label("ERD (%)", color=INK)

    fig.savefig(out_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)


def plot_group_timecourse(times, on_median, off_median, on_participants,
                          off_participants, out_path):
    """One panel: ON vs OFF grand-median ERD time course, participant traces
    shown faintly behind it."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)

    for line in on_participants:
        ax.plot(times, line, color=cfg.COLOR_ON, lw=0.6, alpha=0.18)
    for line in off_participants:
        ax.plot(times, line, color=cfg.COLOR_OFF, lw=0.6, alpha=0.18)

    ax.axvspan(cfg.BASELINE[0], cfg.BASELINE[1], color=GRID, alpha=0.6,
              label="baseline")
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=INK_SOFT, lw=1, ls="--")

    ax.plot(times, on_median, color=cfg.COLOR_ON, lw=2.4, label="Stim ON")
    ax.plot(times, off_median, color=cfg.COLOR_OFF, lw=2.4, label="Stim OFF")

    ax.set_xlabel("Time from movement onset (s)")
    ax.set_ylabel("ERD (%)")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
