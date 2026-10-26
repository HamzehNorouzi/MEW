# -*- coding: utf-8 -*-
"""
Every setting for the Hypothesis 4 pipeline (sustained beta power during gait).
Edit this file, not the scripts.

H4 is a different analysis from H2 and H3. There is no time-frequency
decomposition, no baseline and no time course: each gait epoch gives one Welch
power spectrum, specparam removes the aperiodic component, and beta power is the
area under the flattened spectrum between 14 and 35 Hz.

Because the specparam fit screen is z-scored ACROSS ALL PARTICIPANTS (section
3.5), the pipeline is two-pass: h4_extract.py writes one row per epoch, then
h4_group.py applies the screen and produces the participant-level values.
"""

import numpy as np

# ---- Paths -------------------------------------------------------------------
DERIVATIVES_DIR = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/derivatives"

# Must contain one row per participant, an id column, and "Side of Stimulator".
PARTICIPANT_INFO = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/participants_info.csv"

# Everything this pipeline writes goes here.
OUTPUT_DIR = "/home/hamzeh/Documents/MEAOW_project/hypothesis_4"

# Column names in participant_info.csv.
ID_COLUMN = "Participant ID"
SIDE_COLUMN = "Side of stimulator"

# ---- Markers -----------------------------------------------------------------
# Four-digit trigger codes.
#   digit 1 : condition   1 = OFF, 2 = ON
#   digit 3 : direction   1 = outward, 2 = return
#   digit 4 : phase       3 = gait onset, 4 = gait end
# Every other annotation in the file is ignored.
#
# A gait epoch runs from the onset marker to the end marker of the same
# condition and the same direction, consecutive in the target-marker stream.
ONSET_CODES = {"ON": ["2213", "2223"], "OFF": ["1213", "1223"]}
END_CODES = {"ON": ["2214", "2224"], "OFF": ["1214", "1224"]}

CONDITION_IDS = {"OFF": 1, "ON": 2}

# A participant needs at least this many surviving trials in each condition.
MIN_TRIALS = 30

# Gait epochs outside these bounds are treated as interrupted trials: the walk
# was not completed normally, so the epoch does not describe steady walking.
# They are dropped and counted in the log.
MIN_DURATION_S = 3
MAX_DURATION_S = 10.0


# ---- Segmentation ------------------------------------------------------------
# Each retained gait epoch is cut into consecutive segments of this length.
# The segments serve two purposes, and they are the reason the pipeline works at
# all with walks of different durations:
#
#   AutoReject needs equal-length epochs, which variable-length walks cannot
#   provide. It judges the segments.
#
#   Welch's frequency resolution is sfreq / n_fft, so an epoch-length window
#   would give every walk a different frequency axis and no two spectra could be
#   compared. With n_fft fixed at the segment length, every trial lands on the
#   same grid.
#
# A remainder shorter than one full segment at the end of a walk is not used, so
# every segment is exactly this long.
SEGMENT_S = 2.0

# Overlap between consecutive segments, as a fraction of SEGMENT_S. 0.5 is the
# Welch default and gives smoother spectra; 0.0 keeps segments independent.
SEGMENT_OVERLAP = 0.5


# ---- Spectra -----------------------------------------------------------------
# PSD is computed per segment with n_fft = n_per_seg = SEGMENT_S, then the
# segments of a trial are combined into ONE spectrum for that trial.
PSD_FMIN, PSD_FMAX = 1.0, 100.0

# Beta band the area under the flattened spectrum is taken over.
BETA_BAND = (14.0, 35.0)


# ---- Specparam ---------------------------------------------------------------
# Exactly the settings section 3.5 specifies, which the protocol says to apply
# to gait epochs as well as to resting state.
SPECPARAM_FREQ_RANGE = (1.0, 100.0)
SPECPARAM_PEAK_WIDTH_LIMITS = (0.0, 12.0)
SPECPARAM_PEAK_THRESHOLD = 1.0
SPECPARAM_APERIODIC_MODE = "knee"
SPECPARAM_MAX_N_PEAKS = 7
SPECPARAM_MIN_PEAK_HEIGHT = 0.1

# Goodness-of-fit metric that gets z-scored. "error" is the fit error, where a
# HIGH value is a poor fit, so the screen drops z > +2.
SPECPARAM_FIT_METRIC = "error"
FIT_Z_THRESHOLD = 2.0


# ---- Regions of interest -----------------------------------------------------
# Spectra are computed for both ROIs; the contralateral one (opposite the
# stimulator) is the one used for the statistics.
ROI_LEFT = ["C1", "C3", "C5", "CP1", "CP3", "CP5"]
ROI_RIGHT = ["C2", "C4", "C6", "CP2", "CP4", "CP6"]


# ---- AutoReject --------------------------------------------------------------
# AutoReject judges the SEGMENT_S segments, both conditions pooled so the
# threshold cannot depend on condition. Only the drop log is used; no channel
# interpolation is carried into the spectra.
#
# A flagged segment is simply excluded from its trial's spectrum. A trial is
# then dropped outright only when too little of it survives:
#
#   AR_MIN_CLEAN_SEGMENTS  a trial needs at least this many clean segments,
#                          otherwise its spectrum rests on too little data.
#   AR_MAX_FLAGGED_FRACTION  and no more than this fraction of its segments may
#                          be flagged. 1.0 disables the rule and lets the
#                          minimum-count requirement decide alone.
#
# At a ~3.9 s median walk most trials give only 1-3 segments, so these two
# settings are what actually determine rejection; a fraction threshold on its
# own would be near-meaningless at that length.
AR_MIN_CLEAN_SEGMENTS = 2
AR_MAX_FLAGGED_FRACTION = 1.0

AR_N_INTERPOLATE = [1, 4, 8]
AR_CONSENSUS = np.linspace(0.2, 1.0, 5)
AR_CV = 4
RANDOM_STATE = 313


# ---- Averaging ---------------------------------------------------------------
# Every collapsing step goes through this: across the ROI channels, across the
# clean segments of a trial, and across trials within a condition.
#
# "median" is the project default and the robust choice for mobile EEG recorded
# during walking. "mean" is available for comparison and tags its outputs so
# they cannot overwrite the median ones.
AVERAGING = "median"   # "median" | "mean"


# ---- Plots -------------------------------------------------------------------
COLOR_ON = "#2a78d6"
COLOR_OFF = "#eb6834"
PLOT_DPI = 1200

N_JOBS = -1
