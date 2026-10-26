"""
h1_inspect.py
--------------
Step-by-step inspection of the H1 pipeline for ONE participant.

This does not change anything. It calls the same functions as
h1_extract.py, prints what happens at each step, and saves figures so the
block marking, the concatenation, the epoch counts and the AUC values can
all be eyeballed before the full run.

Usage:
    python h1_inspect.py hx_01

Figures are written to  <OUTPUT_DIR>/inspect_<subject>/

NOTE: the specparam goodness-of-fit screen (z > 2) is a GROUP-level step
and cannot be applied to a single participant. The AUC values shown here
are therefore before that screen. Everything else matches the pipeline.
"""

import os
import sys
import glob
import traceback

# Unbuffered output, so nothing is lost if the script dies part way through.
print("h1_inspect.py starting...", flush=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")            # save files, do not open a window
import matplotlib.pyplot as plt

import mne
from fooof import FOOOF

import h1_config as cfg
import h1_extract as ex


# ---------------------------------------------------------------------------
# Appearance. Two categorical hues, validated for colour-vision deficiency
# (worst adjacent pair dE 24.7 under protanopia).
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
COLOUR = {"ON": "#2a78d6", "OFF": "#eb6834"}
INK = "#1a1a19"
INK_SOFT = "#6b6659"
GRID = "#e3dfd7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "figure.dpi": 130,
})

SESSIONS = ["pre_task_rest", "post_task_rest"]
CONDITIONS = ["ON", "OFF"]


def banner(text):
    print(flush=True)
    print("=" * 74, flush=True)
    print(text, flush=True)
    print("=" * 74, flush=True)


def step(number, text):
    print(flush=True)
    print("-" * 74, flush=True)
    print("STEP %s -- %s" % (number, text), flush=True)
    print("-" * 74, flush=True)


def preflight(subject):
    """
    Check that everything this script needs actually exists, and say
    exactly what is missing. Returns True if it is safe to continue.
    """
    banner("PREFLIGHT")

    print("  working directory : %s" % os.getcwd())
    print("  python            : %s" % sys.version.split()[0])

    ok = True

    checks = [
        ("derivatives folder", cfg.DERIVATIVES_DIR, True),
        ("contralaterality CSV", cfg.CONTRALATERALITY_CSV, False),
        ("stage 0 manifest", getattr(cfg, "MANIFEST_CSV", None), False),
    ]
    for name, path, is_directory in checks:
        if path is None:
            print("  %-22s : not configured" % name)
            continue
        absolute = os.path.abspath(path)
        exists = os.path.isdir(absolute) if is_directory \
            else os.path.isfile(absolute)
        print("  %-22s : %s   %s"
              % (name, "FOUND    " if exists else "MISSING  ", absolute))
        if not exists and name != "stage 0 manifest":
            ok = False

    folder = os.path.join(cfg.DERIVATIVES_DIR, subject)
    absolute_folder = os.path.abspath(folder)
    if not os.path.isdir(absolute_folder):
        print("  subject folder         : MISSING   %s" % absolute_folder)
        available = sorted(os.path.basename(p) for p in
                           glob.glob(os.path.join(cfg.DERIVATIVES_DIR, "hx_*"))
                           if os.path.isdir(p))
        print("  subject folders present: %s"
              % (", ".join(available) if available else "(none)"))
        ok = False
    else:
        print("  subject folder         : FOUND     %s" % absolute_folder)
        for session in SESSIONS:
            fif = os.path.join(folder,
                               "%s_%s_clean-raw.fif" % (subject, session))
            print("      %-16s : %s"
                  % (session, "FOUND" if os.path.isfile(fif) else "MISSING"))
            if not os.path.isfile(fif):
                ok = False
        contents = sorted(os.listdir(absolute_folder))
        print("      folder contains  : %s"
              % (", ".join(contents) if contents else "(empty)"))

    if not ok:
        print()
        print("  Cannot continue. Fix the MISSING entries above, or correct")
        print("  the paths at the top of h1_config.py, then re-run.")

    print(flush=True)
    return ok


# ===========================================================================
# Step 1: what is in the file
# ===========================================================================

def show_annotations(raw, session):
    """Print every distinct annotation description and how often it occurs."""
    counts = {}
    for annot in raw.annotations:
        text = str(annot["description"])
        counts[text] = counts.get(text, 0) + 1

    print("  recording length : %.2f s" % raw.times[-1])
    print("  first_time       : %.2f s  (annotation onsets are shifted by this)"
          % raw.first_time)
    print("  channels         : %d" % len(raw.ch_names))
    print("  distinct annotation descriptions:")
    for text in sorted(counts):
        print("      %-40s x %d" % ("'%s'" % text, counts[text]))


# ===========================================================================
# Step 2: blocks, drawn on a timeline
# ===========================================================================

def collect_markers(raw, condition, logger, label):
    """Marker onsets for one condition, corrected for first_time."""
    all_descriptions = sorted({str(a["description"]) for a in raw.annotations})
    wanted = ex.select_descriptions(all_descriptions, condition, logger, label)

    markers = []
    for annot in raw.annotations:
        if str(annot["description"]) in wanted:
            markers.append(float(annot["onset"]) - float(raw.first_time))
    markers.sort()
    return markers


def plot_block_timeline(raw_by_session, markers_by_key, blocks_by_key,
                        subject, out_dir):
    """
    One row per session per condition. The grey bar is the recording, the
    ticks are the markers, the filled spans are the accepted blocks.
    """
    rows = []
    for session in SESSIONS:
        for condition in CONDITIONS:
            rows.append((session, condition))

    figure, axes = plt.subplots(len(rows), 1, figsize=(11, 1.5 * len(rows)),
                                sharex=True)

    for axis, (session, condition) in zip(axes, rows):
        key = (session, condition)
        raw = raw_by_session[session]
        markers = markers_by_key[key]
        blocks = blocks_by_key[key]
        colour = COLOUR[condition]

        # the recording
        axis.axhspan(-0.28, 0.28, xmin=0, xmax=1, color="#efece6", zorder=0)

        # accepted blocks
        for index, (start, end) in enumerate(blocks):
            axis.axvspan(start, end, ymin=0.18, ymax=0.82,
                         color=colour, alpha=0.85, zorder=2)
            axis.text((start + end) / 2, 0.0,
                      "%.0f s" % (end - start),
                      ha="center", va="center", color="white",
                      fontsize=9, fontweight="bold", zorder=3)

        # every marker
        for onset in markers:
            axis.plot([onset, onset], [-0.42, 0.42], color=INK,
                      linewidth=1.2, zorder=4)

        axis.set_xlim(0, raw.times[-1])
        axis.set_ylim(-0.6, 0.6)
        axis.set_yticks([])
        axis.grid(False)
        axis.set_ylabel("%s\n%s" % (session.replace("_task_rest", ""), condition),
                        rotation=0, ha="right", va="center", fontsize=9)

        summary = "%d markers, %d block(s) accepted" % (len(markers), len(blocks))
        # Placed above the row, not inside it, so it cannot sit on a block.
        axis.set_title(summary, loc="right", fontsize=8.5, color=INK_SOFT,
                       pad=3)

    axes[-1].set_xlabel("time in the cleaned recording (s)")
    figure.suptitle("%s  -  block marking  (vertical lines = annotations, "
                    "filled = accepted block)" % subject,
                    fontsize=11.5, y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(out_dir, "1_block_timeline.png")
    figure.savefig(path)
    plt.close(figure)
    return path


# ===========================================================================
# Step 4: epoch counts
# ===========================================================================

def plot_epoch_counts(counts, subject, out_dir):
    """
    Grouped bars: how many epochs each cell produced and how many AutoReject
    kept. Retained is drawn on top of total, so the gap is the loss.
    """
    labels = []
    totals = []
    retained = []
    colours = []

    for session in SESSIONS:
        for condition in CONDITIONS:
            key = (session, condition)
            labels.append("%s\n%s" % (session.replace("_task_rest", ""),
                                      condition))
            totals.append(counts[key]["total"])
            retained.append(counts[key]["retained"])
            colours.append(COLOUR[condition])

    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7.5, 4.2))

    axis.bar(positions, totals, width=0.62, color="#e6e2da",
             label="dropped by AutoReject")
    axis.bar(positions, retained, width=0.62, color=colours,
             label="retained")

    for x, (total, keep) in enumerate(zip(totals, retained)):
        axis.text(x, keep / 2, str(keep), ha="center", va="center",
                  color="white", fontweight="bold", fontsize=10)
        if total > keep:
            axis.text(x, total + 0.25, "%d dropped" % (total - keep),
                      ha="center", va="bottom", fontsize=8.5, color=INK_SOFT)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("10 s epochs")
    axis.set_ylim(0, max(totals) * 1.25 if max(totals) > 0 else 1)
    axis.grid(axis="x", visible=False)
    axis.set_title("%s  -  epochs after concatenating blocks" % subject,
                   fontsize=11.5, loc="left")

    # A legend is needed because two things are encoded by fill.
    handles = [plt.Rectangle((0, 0), 1, 1, color="#e6e2da"),
               plt.Rectangle((0, 0), 1, 1, color=COLOUR["ON"]),
               plt.Rectangle((0, 0), 1, 1, color=COLOUR["OFF"])]
    axis.legend(handles, ["total produced", "retained (ON)", "retained (OFF)"],
                frameon=False, fontsize=9, loc="upper right")

    figure.tight_layout()
    path = os.path.join(out_dir, "2_epoch_counts.png")
    figure.savefig(path)
    plt.close(figure)
    return path


# ===========================================================================
# Step 6: one spectrum per cell, with the aperiodic fit
# ===========================================================================

def plot_example_spectra(spectra_by_key, subject, out_dir):
    """
    Four panels. Each shows the observed spectrum and the fitted aperiodic
    component on the left axis, and the flattened spectrum below, with the
    2-3 Hz area shaded. This is the quantity H1 is built on.
    """
    # Each cell gets TWO stacked axes: the observed spectrum with its
    # aperiodic fit on top, and the flattened spectrum below on its own
    # scale. Drawing the flattened trace on the same axis as the raw
    # spectrum squashes it flat against the bottom and hides the 2-3 Hz
    # bump entirely, which is the one thing this figure exists to show.
    figure = plt.figure(figsize=(11.5, 8.6))
    grid = figure.add_gridspec(4, 2, height_ratios=[2, 1, 2, 1],
                               hspace=0.45, wspace=0.22)

    panel = 0
    for session in SESSIONS:
        for condition in CONDITIONS:
            row_block = 0 if session == "pre_task_rest" else 2
            column = 0 if condition == "ON" else 1
            axis = figure.add_subplot(grid[row_block, column])
            axis_flat = figure.add_subplot(grid[row_block + 1, column],
                                           sharex=axis)
            panel = panel + 1
            key = (session, condition)
            colour = COLOUR[condition]

            entry = spectra_by_key.get(key)
            if entry is None or entry["spectra"] is None or \
                    entry["spectra"].shape[0] == 0:
                axis.text(0.5, 0.5, "no epochs", ha="center", va="center",
                          transform=axis.transAxes, color=INK_SOFT)
                axis.set_title("%s / %s" % (session.replace("_task_rest", ""),
                                            condition), fontsize=10, loc="left")
                axis_flat.set_xscale("log")
                continue

            freqs = entry["freqs"]
            spectrum = entry["spectra"][0, :]     # the first epoch

            fm = FOOOF(
                peak_width_limits=cfg.FOOOF_SETTINGS["peak_width_limits"],
                max_n_peaks=cfg.FOOOF_SETTINGS["max_n_peaks"],
                min_peak_height=cfg.FOOOF_SETTINGS["min_peak_height"],
                peak_threshold=cfg.FOOOF_SETTINGS["peak_threshold"],
                aperiodic_mode=cfg.FOOOF_SETTINGS["aperiodic_mode"],
                verbose=False,
            )
            fm.fit(freqs, spectrum, cfg.FOOOF_FIT_RANGE)

            fit_freqs = fm.freqs
            observed = fm.power_spectrum
            aperiodic = fm._ap_fit
            flattened = observed - aperiodic

            low, high = cfg.AUC_BAND
            in_band = (fit_freqs >= low) & (fit_freqs <= high)
            auc = ex.area_under_curve(flattened[in_band], fit_freqs[in_band])

            # top: what came out of Welch, and what FOOOF called aperiodic
            axis.plot(fit_freqs, observed, color=INK, linewidth=1.8,
                      label="observed")
            axis.plot(fit_freqs, aperiodic, color=INK_SOFT, linewidth=1.8,
                      linestyle="--", label="aperiodic fit")
            axis.axvspan(low, high, color=colour, alpha=0.10)
            axis.set_xscale("log")
            axis.set_ylabel("log10 power")
            axis.tick_params(labelbottom=False)
            axis.set_title("%s / %s   epoch 1   AUC(2-3 Hz) = %.4f   r2 = %.3f"
                           % (session.replace("_task_rest", ""), condition,
                              auc, fm.r_squared_),
                           fontsize=10, loc="left")
            axis.legend(frameon=False, fontsize=8.5)

            # bottom: the flattened spectrum on its own scale, with the
            # shaded area that becomes the AUC
            axis_flat.axhline(0, color=GRID, linewidth=1)
            axis_flat.plot(fit_freqs, flattened, color=colour, linewidth=1.8)
            axis_flat.fill_between(fit_freqs[in_band], 0, flattened[in_band],
                                   color=colour, alpha=0.40)
            axis_flat.set_xscale("log")
            axis_flat.set_ylabel("flattened")
            # Only the bottom pair carries the x label, otherwise it collides
            # with the title of the panel underneath.
            if row_block == 2:
                axis_flat.set_xlabel("frequency (Hz)")
            else:
                axis_flat.tick_params(labelbottom=True)

    figure.suptitle("%s  -  spectral parameterisation, one example epoch "
                    "per cell   (shaded = the 2-3 Hz area that becomes the "
                    "AUC)" % subject, fontsize=11.5, y=0.985)
    path = os.path.join(out_dir, "3_example_spectra.png")
    figure.savefig(path)
    plt.close(figure)
    return path


# ===========================================================================
# Step 7: AUC per epoch
# ===========================================================================

def plot_auc(auc_by_key, subject, out_dir):
    """
    Every epoch's 2-3 Hz AUC, one column per cell, with the mean marked.
    The right panel shows the two change scores this participant contributes.
    """
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})

    labels = []
    positions = []
    means = {}
    position = 0

    for session in SESSIONS:
        for condition in CONDITIONS:
            key = (session, condition)
            values = np.asarray(auc_by_key.get(key, []), dtype=float)
            values = values[np.isfinite(values)]
            colour = COLOUR[condition]

            if len(values) > 0:
                jitter = (np.random.RandomState(0).rand(len(values)) - 0.5) * 0.22
                left.scatter(np.full(len(values), position) + jitter, values,
                             s=34, color=colour, alpha=0.75,
                             edgecolor=SURFACE, linewidth=1.2, zorder=3)
                mean_value = values.mean()
                means[key] = mean_value
                left.plot([position - 0.28, position + 0.28],
                          [mean_value, mean_value],
                          color=INK, linewidth=2.4, zorder=4)
                left.text(position + 0.33, mean_value, "%.3f" % mean_value,
                          va="center", fontsize=9, color=INK)

            labels.append("%s\n%s" % (session.replace("_task_rest", ""),
                                      condition))
            positions.append(position)
            position = position + 1

    left.set_xticks(positions)
    left.set_xticklabels(labels)
    left.set_ylabel("AUC 2-3 Hz (flattened, log10 power)")
    left.set_xlim(-0.6, len(positions) - 0.2)
    left.grid(axis="x", visible=False)
    left.set_title("per-epoch AUC  (bar = mean)", fontsize=11, loc="left")

    # change scores
    bars = []
    bar_colours = []
    bar_labels = []
    for condition in CONDITIONS:
        pre = means.get(("pre_task_rest", condition))
        post = means.get(("post_task_rest", condition))
        if pre is None or post is None:
            continue
        bars.append(post - pre)
        bar_colours.append(COLOUR[condition])
        bar_labels.append("delta %s" % condition)

    if len(bars) > 0:
        y = np.arange(len(bars))
        right.barh(y, bars, height=0.5, color=bar_colours)
        right.set_yticks(y)
        right.set_yticklabels(bar_labels)
        right.axvline(0, color=INK_SOFT, linewidth=1)

        # Offset the value label in POINTS from the bar end, so a negative
        # bar's label never lands on top of the y tick labels.
        for index, value in enumerate(bars):
            right.annotate("%+.4f" % value,
                           xy=(value, index),
                           xytext=(5 if value >= 0 else -5, 0),
                           textcoords="offset points",
                           va="center",
                           ha="left" if value >= 0 else "right",
                           fontsize=9)

        # Room for those labels at both ends.
        span = max(abs(min(bars)), abs(max(bars)), 1e-9)
        right.set_xlim(min(0, min(bars)) - 0.45 * span,
                       max(0, max(bars)) + 0.45 * span)
        right.grid(axis="y", visible=False)
    right.set_xlabel("post minus pre")
    right.set_title("change scores", fontsize=11, loc="left")

    figure.suptitle("%s  -  AUC and change scores  (before the group-level "
                    "fit screen)" % subject, fontsize=11.5, y=0.99)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "4_auc.png")
    figure.savefig(path)
    plt.close(figure)
    return path


# ===========================================================================
# Main
# ===========================================================================

def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "hx_13"

    banner("INSPECTING %s" % subject)

    if not preflight(subject):
        return

    out_dir = os.path.join(cfg.OUTPUT_DIR, "inspect_%s" % subject)
    os.makedirs(out_dir, exist_ok=True)
    print("  figures will be written to: %s" % os.path.abspath(out_dir))

    logger = ex.setup_logging(cfg.OUTPUT_DIR)

    # -- stimulator side ----------------------------------------------------
    known_ids = sorted(
        os.path.basename(path)
        for path in glob.glob(os.path.join(cfg.DERIVATIVES_DIR, "hx_*"))
        if os.path.isdir(path))
    side_table = ex.load_side_table(cfg.CONTRALATERALITY_CSV, logger,
                                    known_ids=known_ids)
    match = side_table[side_table["id"] == subject]
    if len(match) == 0:
        print("  %s is not in %s. Ids present: %s"
              % (subject, cfg.CONTRALATERALITY_CSV,
                 ", ".join(str(v) for v in side_table["id"].head(30))))
        return
    side = match.iloc[0][cfg.SIDE_COLUMN]
    target_roi = cfg.CONTRALATERAL_ROI[side]
    print("  stimulator on the %s hand  ->  target ROI = %s"
          % (side, target_roi))

    folder = os.path.join(cfg.DERIVATIVES_DIR, subject)

    raw_by_session = {}
    markers_by_key = {}
    blocks_by_key = {}
    counts = {}
    spectra_by_key = {}
    auc_by_key = {}

    for session in SESSIONS:
        label = "%s / %s" % (subject, session)
        fif_path = os.path.join(folder,
                                "%s_%s_clean-raw.fif" % (subject, session))

        step("1 (%s)" % session, "load the file and list its annotations")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        raw_by_session[session] = raw
        show_annotations(raw, session)

        print("  sensor positions present : %s"
              % ex.has_channel_positions(raw))
        for roi_name in ["left", "right"]:
            present = [ch for ch in cfg.ROI_CHANNELS[roi_name]
                       if ch in raw.ch_names]
            print("  %-5s ROI channels found   : %d of %d  %s"
                  % (roi_name, len(present),
                     len(cfg.ROI_CHANNELS[roi_name]), present))

        step("2 (%s)" % session, "find the blocks")
        epochs_by_condition = {}
        for condition in CONDITIONS:
            print()
            print("  --- condition %s ---" % condition)
            markers = collect_markers(raw, condition, logger, label)
            blocks = ex.find_blocks(raw, condition, cfg.MIN_BLOCK_DURATION,
                                    logger, label)

            markers_by_key[(session, condition)] = markers
            blocks_by_key[(session, condition)] = blocks

            print("    markers (s) : %s"
                  % ", ".join("%.2f" % m for m in markers))
            print("    accepted %d block(s):" % len(blocks))
            for index, (start, end) in enumerate(blocks, start=1):
                expected = int((end - start) // cfg.EPOCH_LENGTH)
                print("        block %d : %8.2f s -> %8.2f s   "
                      "(%.2f s, about %d epochs)"
                      % (index, start, end, end - start, expected))

            step("3 (%s / %s)" % (session, condition),
                 "cut epochs inside each block, then concatenate")
            epochs = ex.epochs_from_blocks(raw, blocks, condition, logger, label)
            if epochs is None:
                print("    no epochs produced")
                continue

            metadata = epochs.metadata
            for block_number in sorted(metadata["block"].unique()):
                n = int((metadata["block"] == block_number).sum())
                print("        block %s contributed %d epochs" % (block_number, n))
            print("    concatenated total : %d epochs of %.0f s = %.0f s"
                  % (len(epochs), cfg.EPOCH_LENGTH,
                     len(epochs) * cfg.EPOCH_LENGTH))
            epochs_by_condition[condition] = epochs

        if len(epochs_by_condition) < 2:
            print("\n  Cannot continue for this session: a condition is empty.")
            continue

        step("4 (%s)" % session, "AutoReject, fitted on ON + OFF together")
        all_epochs = mne.concatenate_epochs(
            [epochs_by_condition["ON"], epochs_by_condition["OFF"]],
            verbose=False)
        epochs_clean, reject_log = ex.run_autoreject(all_epochs, logger, label)

        before = all_epochs.metadata["condition"].values
        bad = np.array(reject_log.bad_epochs)
        for condition in CONDITIONS:
            is_condition = (before == condition)
            total = int(is_condition.sum())
            dropped = int(np.sum(bad & is_condition))
            counts[(session, condition)] = {
                "total": total,
                "dropped": dropped,
                "retained": total - dropped,
            }
            loss = dropped / total if total > 0 else 1.0
            verdict = "OK" if loss <= cfg.MAX_EPOCH_LOSS else "EXCLUDES PARTICIPANT"
            print("    %-4s : %2d epochs, %2d dropped (%.0f%%), %2d retained   %s"
                  % (condition, total, dropped, 100 * loss, total - dropped,
                     verdict))

        step("5 (%s)" % session, "PSD and FOOOF on the target ROI (%s)"
             % target_roi)
        metadata_clean = epochs_clean.metadata.reset_index(drop=True)
        freqs, spectra = ex.compute_roi_spectra(epochs_clean, target_roi)
        if freqs is None:
            print("    the %s ROI has no usable channels" % target_roi)
            continue

        print("    PSD: %d epochs x %d frequency bins, %.3f to %.3f Hz, "
              "resolution %.3f Hz"
              % (spectra.shape[0], spectra.shape[1], freqs[0], freqs[-1],
                 freqs[1] - freqs[0]))
        band_bins = int(((freqs >= cfg.AUC_BAND[0]) &
                         (freqs <= cfg.AUC_BAND[1])).sum())
        print("    %d frequency bins fall inside the %.0f-%.0f Hz band"
              % (band_bins, cfg.AUC_BAND[0], cfg.AUC_BAND[1]))

        results = ex.fit_all_spectra(freqs, spectra, logger, label)

        for condition in CONDITIONS:
            rows = metadata_clean.index[
                metadata_clean["condition"] == condition].tolist()
            spectra_by_key[(session, condition)] = {
                "freqs": freqs,
                "spectra": spectra[rows, :] if len(rows) > 0 else None,
            }
            values = []
            r_squared = []
            for row in rows:
                result = results[row]
                if result is None:
                    values.append(np.nan)
                else:
                    values.append(result["auc_2_3"])
                    r_squared.append(result["r_squared"])
            auc_by_key[(session, condition)] = values

            finite = [v for v in values if np.isfinite(v)]
            print("    %-4s : %d fits, %d failed, AUC mean %.4f (SD %.4f), "
                  "r2 mean %.3f"
                  % (condition, len(values), len(values) - len(finite),
                     np.mean(finite) if finite else np.nan,
                     np.std(finite, ddof=1) if len(finite) > 1 else np.nan,
                     np.mean(r_squared) if r_squared else np.nan))

    # -- figures ------------------------------------------------------------
    step("6", "figures")
    paths = []
    if len(blocks_by_key) == 4:
        paths.append(plot_block_timeline(raw_by_session, markers_by_key,
                                         blocks_by_key, subject, out_dir))
    if len(counts) == 4:
        paths.append(plot_epoch_counts(counts, subject, out_dir))
    if len(spectra_by_key) > 0:
        paths.append(plot_example_spectra(spectra_by_key, subject, out_dir))
    if len(auc_by_key) > 0:
        paths.append(plot_auc(auc_by_key, subject, out_dir))

    for path in paths:
        print("    saved %s" % path)

    # -- summary ------------------------------------------------------------
    step("7", "summary")
    print("  cell            blocks  epochs  dropped  retained  mean AUC")
    for session in SESSIONS:
        for condition in CONDITIONS:
            key = (session, condition)
            count = counts.get(key, {"total": 0, "dropped": 0, "retained": 0})
            values = [v for v in auc_by_key.get(key, []) if np.isfinite(v)]
            print("  %-14s %5d  %6d  %7d  %8d  %8s"
                  % ("%s %s" % (session.replace("_task_rest", ""), condition),
                     len(blocks_by_key.get(key, [])),
                     count["total"], count["dropped"], count["retained"],
                     "%.4f" % np.mean(values) if values else "n/a"))

    print()
    for condition in CONDITIONS:
        pre = [v for v in auc_by_key.get(("pre_task_rest", condition), [])
               if np.isfinite(v)]
        post = [v for v in auc_by_key.get(("post_task_rest", condition), [])
                if np.isfinite(v)]
        if pre and post:
            print("  delta %-3s = %.4f - %.4f = %+.4f"
                  % (condition, np.mean(post), np.mean(pre),
                     np.mean(post) - np.mean(pre)))

    print()
    print("  Reminder: the group-level fit screen (z > 2) is NOT applied here,")
    print("  so these AUC means differ slightly from the final pipeline values.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never fail silently. Print the whole traceback and exit non-zero.
        print()
        print("!" * 74)
        print("h1_inspect.py stopped with an error:")
        print("!" * 74)
        traceback.print_exc()
        sys.exit(1)
    print()
    print("h1_inspect.py finished.", flush=True)
