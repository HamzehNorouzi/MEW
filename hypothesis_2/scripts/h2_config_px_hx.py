# -*- coding: utf-8 -*-
"""
Every setting for the Hypothesis 2 pipeline (preparatory beta ERD).
Edit this file, not the scripts.

COHORT-DEPENDENT SETTINGS
-------------------------
Participant folders are named hx_XX or px_XX. h2_extract.py asks which cohort
to load and puts the answer in the H2_COHORT environment variable BEFORE this
module is imported; everything below that depends on the cohort is then derived
from it.

The environment variable, rather than a module-level variable set after import,
is deliberate: h2_extract runs participants through joblib, and the worker
processes import this module from scratch. A value assigned to the module after
import would not survive into them, and the workers would silently fall back to
the hx_ timing while the parent reported px_. Environment variables are
inherited by the workers, so every process derives the same numbers here.

The px_ cohort has a much shorter stand-up to movement-onset interval, so it
gets its own wait threshold and its own epoch, baseline and ERD windows, and it
is never dropped for having too few trials.
"""

import os

import numpy as np

# ---- Cohort ------------------------------------------------------------------
VALID_COHORTS = ("px", "hx")

_requested = os.environ.get("H2_COHORT", "hx").strip().lower().rstrip("_")
if _requested not in VALID_COHORTS:
    raise SystemExit(
        "H2_COHORT is %r, which is not one of: %s. "
        "Run h2_extract.py and answer the cohort question, or set it yourself, "
        "e.g.  H2_COHORT=px python h2_extract.py"
        % (os.environ.get("H2_COHORT"), ", ".join(VALID_COHORTS))
    )

COHORT = _requested                 # "hx" or "px"
SUBJECT_PREFIX = COHORT + "_"       # "hx_" or "px_" -- used to find folders

# Appended to every output filename so a px_ run cannot overwrite the hx_
# results the statistics are built on. Empty for hx_, which keeps the existing
# filenames (h2_erd.csv) exactly as they were.
COHORT_TAG = "" if COHORT == "hx" else "_" + COHORT


# ---- Paths -------------------------------------------------------------------
# Stage-0 output root. Expected layout:
#   DERIVATIVES_DIR/<prefix>XX/<prefix>XX_task_clean-raw.fif
DERIVATIVES_DIR = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/PD_gedai"

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


# ==============================================================================
# COHORT-DEPENDENT TIMING AND EXCLUSION
# ==============================================================================
#
#                      hx_                         px_
#   MIN_WAIT_S         3 s                         1 s
#   MIN_TRIALS         30                          0  (never dropped)
#   TMIN, TMAX         -3.0 .. 0.0                 -1.0 .. 0.0
#   BASELINE           -3.0 .. -2.0  (1000 ms)     -1.0 .. -0.5  (500 ms)
#   ERD_WINDOW         -2.0 .. 0.0   (2000 ms)     -0.5 .. 0.0   (500 ms)
#   TMIN_PAD, TMAX_PAD -4.0 .. 1.0                 -2.0 .. 1.0
#
# The px_ windows are sized to fit inside the 1 s minimum wait: a 500 ms
# baseline immediately followed by a 500 ms ERD window fills exactly the second
# before movement onset. Both cohorts keep a 1 s pad either side of the analysis
# window for the wavelets, which trimmed_times() removes again.
#
# NOTE the px_ AUC integrates 500 ms where the hx_ AUC integrates 2000 ms, so
# the two are NOT on the same scale and must not be pooled in one test without
# rescaling (e.g. dividing by the window length to get mean ERD%).

if COHORT == "px":
    # Trials with a shorter stand-up -> movement-onset interval are dropped.
    MIN_WAIT_S = 1

    # 0 = never drop a px_ participant for trial count, however few survive.
    # A condition with NO surviving trials still yields no AUC; h2_extract
    # reports that case and writes no row for it.
    MIN_TRIALS = 1

    # Epochs (seconds, relative to movement onset)
    TMIN_PAD, TMAX_PAD = -2.0, 1.0   # padded epochs, used for the TFR only
    TMIN, TMAX = -1.0, 0.0           # the analysis window
    BASELINE = (-1.0, -0.5)          # standing period, used as the trial baseline
    ERD_WINDOW = (-0.5, 0.0)         # the window the AUC integrates
else:
    MIN_WAIT_S = 1
    MIN_TRIALS = 1

    TMIN_PAD, TMAX_PAD = -4.0, 1.0
    TMIN, TMAX = -3.0, 0.0
    BASELINE = (-3.0, -2.0)
    ERD_WINDOW = (-2.0, 0.0)

# Logging only. A px_ participant kept with fewer than this many trials is
# called out in h2_run.log, so "never dropped" does not become "never noticed".
LOW_TRIAL_WARNING = 30


# ---- Time-frequency ----------------------------------------------------------
FREQS = np.arange(14.0, 36.0, 1.0)   # 14-35 Hz, 22 bins
N_CYCLES = FREQS / 2.0               # 500 ms wavelet at every frequency
DECIM = 5                            # 250 Hz -> 50 Hz TFR sampling

# NOTE for px_: N_CYCLES = FREQS / 2 gives a ~500 ms wavelet, which is as long
# as the whole px_ baseline and as long as its ERD window. The two windows are
# adjacent, so each one's estimate is smeared into the other and the ERD is
# measured conservatively. Halving the cycles (FREQS / 4, a ~250 ms wavelet)
# would separate them at the cost of frequency resolution. Left at the project
# default here because it is a shared analysis parameter -- change it
# deliberately, not by accident:
#   if COHORT == "px":
#       N_CYCLES = FREQS / 4.0

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
