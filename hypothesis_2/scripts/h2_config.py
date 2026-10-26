# -*- coding: utf-8 -*-
"""
Every setting for the Hypothesis 2 pipeline (preparatory beta ERD).
Edit this file, not the scripts.
"""

import numpy as np

# ---- Paths -------------------------------------------------------------------
# Stage-0 output root. Expected layout: DERIVATIVES_DIR/hx_XX/hx_XX_task_clean-raw.fif
DERIVATIVES_DIR = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/derivatives"

# Must contain one row per participant, an id column, and "Side of Stimulator".
PARTICIPANT_INFO = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/participants_info.csv"

# Everything this pipeline writes goes here.
OUTPUT_DIR = "/home/hamzeh/Documents/MEAOW_project/hypothesis_2"

# Column names in participant_info.csv.
ID_COLUMN = "Participant ID"
SIDE_COLUMN = "Side of stimulator"

# ---- Markers -----------------------------------------------------------------
# Four-digit trigger codes.
#   digit 1 : condition   1 = OFF, 2 = ON
#   digit 3 : direction   1 = outward, 2 = return
#   digit 4 : phase       2 = stand up, 3 = movement onset
# Every other annotation in the file is ignored.
STANDUP_CODES = {"ON": ["2212", "2222"], "OFF": ["1212", "1222"]}
ONSET_CODES = {"ON": ["2213", "2223"], "OFF": ["1213", "1223"]}

CONDITION_IDS = {"OFF": 1, "ON": 2}

# Trials with a shorter stand-up -> movement-onset interval are dropped.
MIN_WAIT_S = 3

# A participant needs at least this many surviving trials in each condition.
MIN_TRIALS = 30


# ---- Epochs (seconds, relative to movement onset) ----------------------------
TMIN_PAD, TMAX_PAD = -4.0, 1.0   # padded epochs, used for the TFR only
TMIN, TMAX = -3.0, 0.0           # the analysis window
BASELINE = (-3.0, -2.0)          # standing period, used as the trial baseline
ERD_WINDOW = (-2.0, 0.0)         # the window the AUC integrates


# ---- Time-frequency ----------------------------------------------------------
FREQS = np.arange(14.0, 36.0, 1.0)   # 14-35 Hz, 22 bins
N_CYCLES = FREQS / 2.0               # 500 ms wavelet at every frequency
DECIM = 5                            # 250 Hz -> 50 Hz TFR sampling

# Moving average on the ERD time course. This is the analysis parameter: it
# shapes the curve the AUC integrates, so it is locked to what was agreed and
# used everywhere the AUC is computed (h2_erd.csv, the R script downstream).
SMOOTH_MS = 400

# Moving average used ONLY for the group TFR image (h2_group_tfr.png / the
# inspection TFR). Lighter than SMOOTH_MS on purpose: the TFR already carries
# the wavelet's own ~500 ms temporal smoothing (n_cycles = freq / 2) and a
# median across trials and channels, so stacking the full 400 ms on top mostly
# just blurs time further without adding robustness. Never touches the AUC.
PLOT_SMOOTH_MS = 100


# ---- Averaging ---------------------------------------------------------------
# How the ERD is collapsed at every stage: the per-trial baseline over time,
# across trials, across the 22 beta bins, across the 6 ROI channels, and across
# participants for the group figures.
#
# "median" is the project default and what the analysis is specified on. It is
# the robust choice for mobile EEG recorded during walking, where a few trials
# carry very large artifacts.
#
# "mean" is available for comparison. Switching it changes the AUC in h2_erd.csv
# and therefore the statistics downstream, so outputs written under "mean" are
# tagged and cannot overwrite the median ones.
AVERAGING = "median"   # "median" | "mean"


# ---- Regions of interest -----------------------------------------------------
# The ROI used is the hemisphere opposite the stimulator.
ROI_LEFT = ["C1", "C3", "C5", "CP1", "CP3", "CP5"]
ROI_RIGHT = ["C2", "C4", "C6", "CP2", "CP4", "CP6"]


# ---- AutoReject --------------------------------------------------------------
AR_N_INTERPOLATE = [1, 4, 8]
AR_CONSENSUS = np.linspace(0.2, 1.0, 5)
AR_CV = 4
RANDOM_STATE = 97


# ---- Inspection plots ---------------------------------------------------------
# Colours for the two conditions, used throughout the pipeline's plots.
COLOR_ON = "#2a78d6"
COLOR_OFF = "#eb6834"
PLOT_DPI = 1200

# TFR colour scale: clipped to this percentile of |ERD %|, both conditions
# combined, rather than the literal max. A handful of outlier pixels (there is
# no AutoReject in the inspection TFR) would otherwise stretch the scale and
# wash out the contrast in the range that actually matters.
TFR_COLOR_PERCENTILE = 98
N_JOBS = -1
