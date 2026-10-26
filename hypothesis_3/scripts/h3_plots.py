# -*- coding: utf-8 -*-
"""
Group-level ERS figures for the Hypothesis 3 pipeline.

Sign convention: the percent change is computed exactly as in H2 and never
flipped, so synchronisation comes out POSITIVE. Red is positive in the TFR and
the post-movement rebound is a peak, not a valley. (H2's ERD is the mirror of
this: negative, blue, a valley.)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import h3_config as cfg

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Diverging blue <-> red, gray at zero. Red = ERS (positive), blue = ERD.
ERS_CMAP = LinearSegmentedColormap.from_list("ers", ["#2a78d6", "#f0efec", "#e34948"])


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

    Colour range is clipped to a percentile of |ERS %| rather than the literal
    max, so a few outlier pixels don't stretch the scale and flatten the
    contrast in the range that actually carries the rebound. Values beyond the
    clip still show, fully saturated at the colormap's end.
    """
    combined = np.concatenate([np.abs(tfr_on).ravel(), np.abs(tfr_off).ravel()])
    vmax = np.percentile(combined, cfg.TFR_COLOR_PERCENTILE)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True,
                             constrained_layout=True, facecolor=SURFACE)
    for ax, data, label in zip(axes, [tfr_on, tfr_off], ["Stim ON", "Stim OFF"]):
        im = ax.pcolormesh(times, freqs, data, cmap=ERS_CMAP, norm=norm,
                           shading="gouraud")
        ax.axvline(0, color=INK_SOFT, lw=1, ls="--")
        ax.set_title(label, color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time from movement stop (s)")
        _style_axis(ax)
    axes[0].set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(im, ax=axes, label="ERS (%)", shrink=0.9)
    cbar.ax.tick_params(colors=INK_SOFT, labelsize=9)
    cbar.set_label("ERS (%)", color=INK)

    fig.savefig(out_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)


def plot_group_timecourse(times, on_median, off_median, on_participants,
                          off_participants, out_path):
    """One panel: ON vs OFF grand-median ERS time course, participant traces
    shown faintly behind it.

    There is no baseline shading here, unlike the H2 figure: the baseline is
    locked to movement onset and so lies outside this window entirely.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)

    for line in on_participants:
        ax.plot(times, line, color=cfg.COLOR_ON, lw=0.6, alpha=0.18)
    for line in off_participants:
        ax.plot(times, line, color=cfg.COLOR_OFF, lw=0.6, alpha=0.18)

    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=INK_SOFT, lw=1, ls="--")

    ax.plot(times, on_median, color=cfg.COLOR_ON, lw=2.4, label="Stim ON")
    ax.plot(times, off_median, color=cfg.COLOR_OFF, lw=2.4, label="Stim OFF")

    ax.set_xlabel("Time from movement stop (s)")
    ax.set_ylabel("ERS (%)")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
