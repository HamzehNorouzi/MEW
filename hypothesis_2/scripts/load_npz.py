# -*- coding: utf-8 -*-
"""
Re-plot a group TFR .npz (h2_group_tfr.npz or h3_group_tfr.npz) without
needing the rest of the pipeline installed. Useful for a quick look, or for
redrawing the figure with different smoothing/colour choices later.

Array shapes inside the .npz:
    participants   (n_participants,)                 str
    conditions     (n_conditions,)                    str, e.g. ["ON","OFF"]
    times          (n_times,)                         s
    freqs          (n_freqs,)                         Hz
    tfr            (n_participants, n_conditions, n_freqs, n_times)   % change
    timecourse     (n_participants, n_conditions, n_times)            % change

Usage:
    python plot_group_npz.py h2_group_tfr.npz
    python plot_group_npz.py h3_group_tfr.npz --averaging mean
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

COLOR_ON = "#2a78d6"
COLOR_OFF = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
TFR_COLOR_PERCENTILE = 98

# Diverging TFR colormap matching h2_plots.py / h3_plots.py.
from matplotlib.colors import LinearSegmentedColormap
ERD_CMAP = LinearSegmentedColormap.from_list(
    "erd", ["#2a78d6", "#f0efec", "#e34948"])


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_tfr(times, freqs, tfr_on, tfr_off, out_path, x_label):
    combined = np.concatenate([np.abs(tfr_on).ravel(), np.abs(tfr_off).ravel()])
    vmax = np.percentile(combined, TFR_COLOR_PERCENTILE)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True,
                             constrained_layout=True, facecolor=SURFACE)
    im = None
    for ax, data, label in zip(axes, [tfr_on, tfr_off], ["Stim ON", "Stim OFF"]):
        im = ax.pcolormesh(times, freqs, data, cmap=ERD_CMAP, norm=norm,
                           shading="gouraud")
        ax.axvline(0, color=INK_SOFT, lw=1, ls="--")
        ax.set_title(label, color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label)
        _style_axis(ax)
    axes[0].set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(im, ax=axes, label="% change", shrink=0.9)
    cbar.ax.tick_params(colors=INK_SOFT, labelsize=9)
    cbar.set_label("% change", color=INK)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


def plot_timecourse(times, on_median, off_median, on_traces, off_traces,
                    out_path, x_label):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)

    for line in on_traces:
        ax.plot(times, line, color=COLOR_ON, lw=0.6, alpha=0.18)
    for line in off_traces:
        ax.plot(times, line, color=COLOR_OFF, lw=0.6, alpha=0.18)

    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=INK_SOFT, lw=1, ls="--")

    ax.plot(times, on_median, color=COLOR_ON, lw=2.4, label="Stim ON")
    ax.plot(times, off_median, color=COLOR_OFF, lw=2.4, label="Stim OFF")

    ax.set_xlabel(x_label)
    ax.set_ylabel("% change")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="h2_group_tfr.npz or h3_group_tfr.npz")
    parser.add_argument("--averaging", choices=["median", "mean"], default="median",
                        help="how to collapse across participants (default: median, "
                             "matching the project default)")
    args = parser.parse_args()

    collapse = np.mean if args.averaging == "mean" else np.median

    with np.load(args.npz_path, allow_pickle=False) as data:
        participants = data["participants"]
        conditions = list(data["conditions"])
        times = data["times"]
        freqs = data["freqs"]
        tfr = data["tfr"]                 # (participants, conditions, freqs, times)
        timecourse = data["timecourse"]   # (participants, conditions, times)

    on, off = conditions.index("ON"), conditions.index("OFF")
    print("%d participants, conditions=%s" % (len(participants), conditions))

    # h3's stop-relative files use a different x-axis label than h2's
    # onset-relative ones; guessed from the filename, override by editing below
    # if you renamed the file.
    is_h3 = "h3" in args.npz_path.lower()
    x_label = ("Time from movement stop (s)" if is_h3
              else "Time from movement onset (s)")

    stem = args.npz_path.rsplit(".", 1)[0]

    tfr_on = collapse(tfr[:, on], axis=0)     # (freqs, times)
    tfr_off = collapse(tfr[:, off], axis=0)
    plot_tfr(times, freqs, tfr_on, tfr_off, stem + "_tfr_replot.png", x_label)

    tc_on_median = collapse(timecourse[:, on], axis=0)   # (times,)
    tc_off_median = collapse(timecourse[:, off], axis=0)
    plot_timecourse(times, tc_on_median, tc_off_median,
                    timecourse[:, on], timecourse[:, off],
                    stem + "_timecourse_replot.png", x_label)


if __name__ == "__main__":
    main()