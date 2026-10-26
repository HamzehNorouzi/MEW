# -*- coding: utf-8 -*-
"""
Every setting for the Hypothesis 3 pipeline (post-movement beta ERS).
Edit this file, not the scripts.

H3 differs from H2 in one structural way. The analysis window is locked to
MOVEMENT STOP, but the protocol takes the baseline from the standing period
before MOVEMENT ONSET ("a 1-second baseline window during the standing period
prior to movement onset, -3,000 to -2,000 ms relative to the start threshold").
Those are two different time origins with a walk of variable length between
them, so every trial needs two epochs: a baseline epoch cut at the onset
marker and an analysis epoch cut at the stop marker.
"""

import numpy as np

# ---- Paths -------------------------------------------------------------------
# Stage-0 output root. Expected layout: DERIVATIVES_DIR/hx_XX/hx_XX_task_clean-raw.fif
DERIVATIVES_DIR = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/derivatives"

# Must contain one row per participant, an id column, and "Side of Stimulator".
PARTICIPANT_INFO = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/participants_info.csv"

# Everything this pipeline writes goes here.
OUTPUT_DIR = "/home/hamzeh/Documents/MEAOW_project/hypothesis_3"

# Column names in participant_info.csv.
ID_COLUMN = "Participant ID"
SIDE_COLUMN = "Side of stimulator"

# ---- Markers -----------------------------------------------------------------
# Four-digit trigger codes.
#   digit 1 : condition   1 = OFF, 2 = ON
#   digit 3 : direction   1 = outward, 2 = return
#   digit 4 : phase       2 = standing/waiting period, 3 = movement onset,
#                         4 = movement stop
# Every other annotation in the file is ignored.
#
# A valid H3 trial is the triple standing -> onset -> stop, consecutive in the
# target-marker stream and all the same condition and direction. The onset
# marker is not t = 0 here, but it is still required: it defines the baseline
# window and the waiting-period rule.
STANDING_CODES = {"ON": ["2212", "2222"], "OFF": ["1212", "1222"]}
ONSET_CODES = {"ON": ["2213", "2223"], "OFF": ["1213", "1223"]}
STOP_CODES = {"ON": ["2214", "2224"], "OFF": ["1214", "1224"]}

CONDITION_IDS = {"OFF": 1, "ON": 2}

# Trials with a shorter standing -> movement-onset interval are dropped, so the
# baseline window always sits inside the waiting period.
MIN_WAIT_S = 2.8

# A participant needs at least this many surviving trials in each condition.
MIN_TRIALS = 30


# ---- Epochs (seconds) --------------------------------------------------------
# Analysis epoch, relative to MOVEMENT STOP.
# TMIN_PAD reaches back to -2.0 so that the exploratory stop-locked baseline
# below (-1 to 0 s) has a clean second of data in front of it for the wavelets.
TMIN_PAD, TMAX_PAD = -2.0, 3.0   # padded, used for the TFR only
TMIN, TMAX = 0.0, 2.0            # the analysis window, also the plotted window
ERS_WINDOW = (0.0, 2.0)          # the window the AUC integrates

# Baseline epoch, relative to MOVEMENT ONSET.
BASE_TMIN_PAD, BASE_TMAX_PAD = -4.0, -1.0   # padded, used for the TFR only
BASELINE = (-3.0, -2.0)                     # standing period, the trial baseline


# ---- Which baseline to use ---------------------------------------------------
# "onset"  THE PROTOCOL. BASELINE above, -3 to -2 s relative to movement onset:
#          quiet standing before gait. ERS is then power relative to rest, which
#          is what H3 is defined on and what must be reported as confirmatory.
#
# "stop"   EXPLORATORY ONLY. BASELINE_STOP below, the second immediately before
#          movement stop -- which is the END OF THE WALK, not rest. This measures
#          post-stop beta relative to late-gait beta. The curve necessarily
#          starts near zero because it is divided by the period right before it,
#          so a rise here is not evidence of a rebound that the protocol
#          baseline missed. Never report a "stop" result as the H3 outcome.
BASELINE_MODE = "onset"           # "onset" | "stop"
BASELINE_STOP = (-2.0, 0.0)      # used only when BASELINE_MODE == "stop"


# ---- Time-frequency ----------------------------------------------------------
FREQS = np.arange(14.0, 36.0, 1.0)   # 14-35 Hz, 22 bins
N_CYCLES = FREQS / 2.0               # 500 ms wavelet at every frequency
DECIM = 5                            # 250 Hz -> 50 Hz TFR sampling

# Moving average on the ERS time course. This is the analysis parameter: it
# shapes the curve the AUC integrates, so it is used everywhere the AUC is
# computed.
SMOOTH_MS = 400

# Moving average used ONLY for the group TFR image. Lighter than SMOOTH_MS on
# purpose: the TFR already carries the wavelet's own ~500 ms temporal smoothing
# and a median across trials and channels. Never touches the AUC.
PLOT_SMOOTH_MS = 100


# ---- Averaging ---------------------------------------------------------------
# How the ERS is collapsed at every stage: the per-trial baseline over time,
# across trials, across the 22 beta bins, across the 6 ROI channels, and across
# participants for the group figures.
#
# "median" is the project default and what the analysis is specified on. It is
# the robust choice for mobile EEG recorded during walking, where a few trials
# carry very large artifacts.
#
# "mean" is available for comparison. Switching it changes the AUC and therefore
# the statistics downstream, so outputs written under "mean" are tagged and
# cannot overwrite the median ones.
AVERAGING = "median"   # "median" | "mean"


# ---- Regions of interest -----------------------------------------------------
# The ROI used is the hemisphere opposite the stimulator.
ROI_LEFT = ["C1", "C3", "C5", "CP1", "CP3", "CP5"]
ROI_RIGHT = ["C2", "C4", "C6", "CP2", "CP4", "CP6"]


# ---- AutoReject --------------------------------------------------------------
# In H2 the baseline and the analysis window sat in ONE epoch, so a single
# AutoReject fit judged both. H3 splits them across two epochs at two time
# origins, so "the same rule" has to be spelled out.
#
# "both"      The faithful analogue of H2, and the default. A trial is dropped
#             if AutoReject flags EITHER its baseline window or its analysis
#             window, because a noisy baseline is a bad denominator just as
#             surely as a noisy task window is a bad numerator. Costs a second
#             fit, though on ~1 s epochs rather than the 2 s analysis ones, so
#             the total data judged is close to H2's single 3 s fit.
#
# "analysis"  Fit only on the stop-locked analysis window. Faster, but a trial
#             with a corrupted baseline survives and silently distorts the
#             percent change for that trial.
#
# Under BASELINE_MODE = "stop" this setting is ignored: both windows then live
# inside the same stop-locked epoch, so one fit covers them anyway.
AR_SCOPE = "both"      # "both" | "analysis"

AR_N_INTERPOLATE = [1, 4, 8]
AR_CONSENSUS = np.linspace(0.2, 1.0, 5)
AR_CV = 4
RANDOM_STATE = 97


# ---- Plots -------------------------------------------------------------------
# Colours for the two conditions.
COLOR_ON = "#2a78d6"
COLOR_OFF = "#eb6834"
PLOT_DPI = 1200

# TFR colour scale: clipped to this percentile of |ERS %|, both conditions
# combined, rather than the literal max, so a few outlier pixels cannot stretch
# the scale and wash out the contrast in the range that matters.
TFR_COLOR_PERCENTILE = 98
N_JOBS = -1
