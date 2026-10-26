"""
h1_config.py
-------------
All settings for the Hypothesis 1 pipeline live here.

Edit this file, not the analysis scripts.
Every number below is either taken from the protocol (section 3.5 / 3.6)
or from the decisions agreed in the flowchart.
"""

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

# Folder produced by stage 0. Must contain the hx_XX subfolders.
DERIVATIVES_DIR = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/derivatives"

# Where this pipeline writes its own outputs.
OUTPUT_DIR = "/home/hamzeh/Documents/MEAOW_project/hypothesis_1"

# CSV holding the stimulator side for each participant.
CONTRALATERALITY_CSV = "/home/hamzeh/Documents/MEAOW_project/hypothesis_1/participants_info.csv"

# The stage 0 manifest. This is the preferred source for the rejected IC
# count and for cross-checking the block detection, because it is written
# by stage 0 itself. Set to None to fall back to the per-session JSON files.
MANIFEST_CSV = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/derivatives/stage0_manifest.csv"

# The stage 0 console log. Used to read the ICLabel class counts, which are
# not in the manifest.
STAGE0_LOG = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/derivatives/stage0_run.log"

# Column in that CSV holding the participant id (values must match the
# folder names, e.g. "hx_01"). Matching ignores case and stray spaces in
# the header. If this name is not found, the script tries "Participant ID",
# "Subject", and finally a bare "Id" -- in that order, because a survey
# export's "Id" is usually the response number rather than the participant.
ID_COLUMN = "Participant ID"

# Column holding the stimulator side. Values must be "Right" or "Left".
SIDE_COLUMN = "Side of stimulator"

# Optional column giving enrolment order for the sequential analysis.
# If absent, participants are ordered by folder name.
ORDER_COLUMN = "order"

# ---------------------------------------------------------------------------
# Block-order validation
# ---------------------------------------------------------------------------
# A CSV recording which condition each rest block was supposed to be. Set to
# None to use CONTRALATERALITY_CSV (they are usually the same export).
PARTICIPANT_INFO_CSV = None

# The column holding the condition of each block, in chronological order.
# Header matching ignores case, punctuation and extra spaces, so
# "Pre-task, resting-state, block 1" and "Pre task resting state Block 1"
# both match.
BLOCK_COLUMNS = {
    "pre_task_rest": [
        "Pre-task, resting-state, block 1",
        "Pre-task, resting-state, block 2",
        "Pre-task, resting-state, block 3",
        "Pre-task, resting-state, block 4",
    ],
    "post_task_rest": [
        "Post-task, resting-state, Block 1",
        "Post-task, resting-state, Block 2",
        "Post-task, resting-state, Block 3",
        "Post-task, resting-state, Block 4",
    ],
}

# How the CSV values map onto the annotation codes.
#   Stimulation -> ON  -> "Comment/timulus/s1004"
#   Sham        -> OFF -> "Comment/timulus/s1002"
CONDITION_VALUES = {
    "stimulation": "ON",
    "stim": "ON",
    "sham": "OFF",
}

# If False, a block whose condition disagrees with the CSV is kept and only
# logged. Leave True so disagreeing blocks are dropped.
DROP_BLOCKS_FAILING_VALIDATION = True


# ---------------------------------------------------------------------------
# 2. Annotations
# ---------------------------------------------------------------------------

# Each block is marked by TWO identical annotations: one at the start and
# one at the end.
#
# The exact annotation text as it appears in the recording. Matching is
# case-insensitive and ignores surrounding whitespace, but is otherwise an
# EXACT full-string comparison, so a marker is never matched by accident.
# List more than one string per condition if the text varies between files.
#
# NOTE: swap ON and OFF here if the mapping is the other way round.
ANNOTATION_DESCRIPTIONS = {
    "ON": ["Comment/timulus/s1004"],   # eyes_closed_Stim_ON
    "OFF": ["Comment/timulus/s1002"],  # eyes_closed_Stim_OFF
}

# Fallback only. If none of the strings above match anything in a file,
# the script looks for annotations whose LAST group of digits equals the
# code below, and says so in the log. This rescues files where the prefix
# differs (e.g. "Stimulus/S1004" instead of "Comment/timulus/s1004").
ANNOTATION_CODES = {
    "ON": "1004",
    "OFF": "1002",
}

# Set to False to switch the fallback off and require an exact match.
ALLOW_ANNOTATION_CODE_FALLBACK = True

# A block is only kept if it lasts at least this long (seconds).
# The protocol says 100 s, but the blocks are actually ~60 s.
MIN_BLOCK_DURATION = 60


# ---------------------------------------------------------------------------
# 3. Epoching
# ---------------------------------------------------------------------------

EPOCH_LENGTH = 10.0   # seconds, non-overlapping
EPOCH_OVERLAP = 0.0   # seconds

# Participant is excluded if AutoReject drops MORE than this fraction of
# epochs in ANY of the four cells (pre-ON, pre-OFF, post-ON, post-OFF).
MAX_EPOCH_LOSS = 0.50


# ---------------------------------------------------------------------------
# 4. Regions of interest (protocol section 3.5)
# ---------------------------------------------------------------------------

ROI_CHANNELS = {
    "left": ["C1", "C3", "C5", "CP1", "CP3", "CP5"],
    "right": ["C2", "C4", "C6", "CP2", "CP4", "CP6"],
}

# Stimulator side -> target ROI is the CONTRALATERAL hemisphere.
CONTRALATERAL_ROI = {
    "Right": "left",
    "Left": "right",
}


# ---------------------------------------------------------------------------
# 5. Power spectral density
# ---------------------------------------------------------------------------

PSD_FMIN = 2.0
PSD_FMAX = 100.0

# Welch window length in seconds. 4 s gives a 0.25 Hz frequency resolution,
# which puts 5 frequency bins inside the 2-3 Hz band of interest.
WELCH_WINDOW_SEC = 4.0
WELCH_OVERLAP_SEC = 2.0


# ---------------------------------------------------------------------------
# 6. Spectral parameterisation (FOOOF / specparam v1)
# ---------------------------------------------------------------------------

FOOOF_SETTINGS = {
    "peak_width_limits": [0.0, 12.0],
    "max_n_peaks": 7,
    "min_peak_height": 0.1,
    "peak_threshold": 1.0,
    "aperiodic_mode": "knee",
}

FOOOF_FIT_RANGE = [2.0, 100.0]

# Band used for the area under the flattened spectrum.
AUC_BAND = [2.0, 3.0]

# Epochs whose FOOOF fit ERROR z-scores above this value are dropped.
# The z-score is computed across ALL participants (see flowchart panel B).
FIT_ERROR_Z_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# 7. Stage 0 quality check
# ---------------------------------------------------------------------------

TOTAL_ICS = 60          # 64 channels - 1 EOG - 2 mastoids - 1
MAX_IC_REJECT_FRACTION = 0.80

# A participant is excluded if EITHER rest recording has fewer than this
# many components labelled 'brain' by ICLabel. The count is read from the
# stage 0 log line:
#     ICLabel: {'brain': 25, 'eye blink': 4, 'other': 31}
MIN_BRAIN_ICS = 12

# Name of the key in the stage 0 JSON holding the rejected ICs.
# Leave as None to let the script search for it automatically. If the search
# fails, the log lists every key in the file: copy the right one in here.
# A nested key can be given with dots, e.g. "ica.exclude".
# The value may be either a count or a list of component indices.
IC_REJECT_JSON_KEY = None


# ---------------------------------------------------------------------------
# 8. Sequential (early stopping) analysis
# ---------------------------------------------------------------------------

# The protocol's stopping rule begins interim testing at N = 22 and stops
# at N = 39. Lowering SEQUENTIAL_N_MIN makes h1_group.py emit interim rows
# from a smaller sample, which is useful for exploring the data.
#
# WARNING: interim Bayes factors computed below N = 22 are NOT part of the
# pre-registered stopping rule and must not be used to decide whether to
# stop collecting. Any change here is written into h1_run.log so the
# deviation is on the record.
SEQUENTIAL_N_MIN = 22
SEQUENTIAL_N_MAX = 39


# ---------------------------------------------------------------------------
# 9. Misc
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Number of parallel jobs. -1 uses every available core. This is passed to
# AutoReject, to the PSD computation, and to the FOOOF group fit.
N_JOBS = -1

# AutoReject interpolates bad channels, which needs sensor positions.
# A file with no montage cannot be processed, so it is excluded rather than
# analysed with a silently degraded AutoReject. Set to False only if you
# have deliberately decided to accept that.
REQUIRE_CHANNEL_POSITIONS = True
