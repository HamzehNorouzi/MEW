"""
h1_extract.py
--------------
Hypothesis 1, panel A of the flowchart: per-participant extraction.

For every participant folder in DERIVATIVES_DIR this script:
    1. looks up the stimulator side and works out the contralateral ROI
    2. checks the stage 0 ICA rejection count
    3. finds the STIM-ON and STIM-OFF blocks from paired annotations
    4. cuts 10 s epochs inside each block, then pools them per condition
    5. runs AutoReject once per session on ON + OFF epochs together
    6. applies the >50% epoch-loss exclusion rule
    7. computes a Welch PSD, averages it within the left and right ROI
    8. fits FOOOF to every epoch spectrum and stores the 2-3 Hz area
       under the flattened spectrum

Output: h1_epoch_level.csv  (one row per epoch per ROI)
        h1_qc_report.csv
        h1_exclusions.csv
        h1_run.log

Run it with:  python h1_extract.py
"""

import os
import re
import ast
import json
import glob
import logging

import numpy as np
import pandas as pd
import mne
from autoreject import AutoReject
from fooof import FOOOF, FOOOFGroup

import h1_config as cfg


# NumPy renamed trapz to trapezoid in version 2.0. This keeps the script
# working on both old and new NumPy.
if hasattr(np, "trapezoid"):
    area_under_curve = np.trapezoid
else:
    area_under_curve = np.trapz


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(output_dir):
    """Set up a logger that writes to both the screen and an append-only file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "h1_run.log")

    logger = logging.getLogger("h1")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear old handlers if the script is re-run

    file_handler = logging.FileHandler(log_path, mode="a")
    stream_handler = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ===========================================================================
# Helpers
# ===========================================================================

def normalise_header(text):
    """
    Reduce a column header to a comparable form: lower case, punctuation
    and runs of whitespace collapsed to single spaces.

    "Pre-task, resting-state, block 1"  ->  "pre task resting state block 1"
    """
    text = str(text).lower()
    cleaned = []
    for character in text:
        cleaned.append(character if character.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def find_column(columns, candidates, logger, what):
    """
    Find a column by name, ignoring case and surrounding whitespace.

    Exported survey files routinely carry stray spaces in their headers
    (" Side of stimulator ") and inconsistent capitalisation ("Id"), so a
    plain `name in table.columns` test is not reliable.

    `candidates` is tried in order, so the most specific name must come
    first. Returns the ORIGINAL column name, or None.
    """
    lookup = {}
    for column in columns:
        lookup[str(column).strip().lower()] = column

    for candidate in candidates:
        if candidate is None:
            continue
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]

    # Second pass: ignore punctuation and repeated spaces too.
    loose = {}
    for column in columns:
        loose[normalise_header(column)] = column
    for candidate in candidates:
        if candidate is None:
            continue
        key = normalise_header(candidate)
        if key in loose:
            return loose[key]

    return None


def normalise_participant_id(value, known_ids):
    """
    Match a participant id from the CSV to a folder name such as "hx_01".

    Tries the value as written, then a few tolerant variants. Returns the
    matching folder name, or None if nothing matches.
    """
    text = str(value).strip()
    if text in known_ids:
        return text

    lookup = {}
    for known in known_ids:
        lookup[known.strip().lower()] = known

    # case and separator differences: "HX-01", "hx 01", "Hx_01"
    simple = text.lower().replace("-", "_").replace(" ", "_")
    if simple in lookup:
        return lookup[simple]

    # a bare number: 1 or 01 -> hx_01
    digits = "".join(character for character in text if character.isdigit())
    if digits != "" and digits == text.replace(" ", ""):
        candidate = "hx_%02d" % int(digits)
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    # "hx01" with no separator
    if digits != "":
        candidate = "hx_%02d" % int(digits)
        if candidate.lower() in lookup and text.lower().startswith("hx"):
            return lookup[candidate.lower()]

    return None


def load_side_table(csv_path, logger, known_ids=None):
    """
    Read the contralaterality CSV and return a tidy DataFrame with an "id"
    column and the stimulator side.

    Column names are matched case-insensitively and stripped of whitespace.
    The participant column is chosen by preferring specific names over the
    generic "Id", because survey exports often contain a response "Id" that
    has nothing to do with the participant.
    """
    table = pd.read_csv(csv_path)
    original_columns = list(table.columns)
    logger.info("Contralaterality CSV columns: %s",
                ", ".join("'%s'" % c for c in original_columns))

    # -- participant id column ---------------------------------------------
    # Specific names FIRST. Bare "id" is last on purpose.
    id_candidates = [cfg.ID_COLUMN, "participant id", "participant_id",
                     "participantid", "participant", "subject id",
                     "subject_id", "subject", "id"]
    id_col = find_column(original_columns, id_candidates, logger, "participant id")

    if id_col is None:
        raise ValueError(
            "Could not find a participant id column in %s.\n"
            "Columns present: %s\n"
            "Set ID_COLUMN in h1_config.py to the right one."
            % (csv_path, original_columns))

    # Warn if we settled for a generic "Id" while something better exists.
    if str(id_col).strip().lower() == "id":
        better = find_column(original_columns,
                             ["participant id", "participant_id", "participant"],
                             logger, "participant id")
        if better is not None:
            logger.warning("Both '%s' and '%s' exist. Using '%s'. A survey "
                           "export's 'Id' is usually the response number, "
                           "not the participant.", id_col, better, better)
            id_col = better

    # -- side column --------------------------------------------------------
    side_candidates = [cfg.SIDE_COLUMN, "side of stimulator", "side",
                       "stimulator side", "stimulator"]
    side_col = find_column(original_columns, side_candidates, logger, "side")

    if side_col is None:
        # last resort: any header mentioning both "side" and "stim"
        for column in original_columns:
            name = str(column).strip().lower()
            if "side" in name and "stim" in name:
                side_col = column
                break

    if side_col is None:
        raise ValueError(
            "Could not find the stimulator side column in %s.\n"
            "Columns present: %s\n"
            "Set SIDE_COLUMN in h1_config.py to the right one."
            % (csv_path, original_columns))

    logger.info("Using '%s' as the participant id and '%s' as the "
                "stimulator side", id_col, side_col)

    # -- tidy up -------------------------------------------------------------
    table = table[[id_col, side_col]].copy()
    table.columns = ["id", cfg.SIDE_COLUMN]

    table["id"] = table["id"].astype(str).str.strip()
    table[cfg.SIDE_COLUMN] = (table[cfg.SIDE_COLUMN].astype(str)
                              .str.strip().str.capitalize())

    # Drop rows with no id or no side.
    before = len(table)
    table = table[(table["id"] != "") & (table["id"].str.lower() != "nan")]
    table = table[table[cfg.SIDE_COLUMN].str.lower() != "nan"]
    if len(table) < before:
        logger.warning("Dropped %d row(s) from the CSV with a missing id or "
                       "side", before - len(table))

    # -- match ids to folder names ------------------------------------------
    if known_ids is not None and len(known_ids) > 0:
        mapped = []
        unmatched = []
        for value in table["id"]:
            match = normalise_participant_id(value, known_ids)
            if match is None:
                unmatched.append(value)
                mapped.append(value)
            else:
                if match != value:
                    logger.info("CSV id '%s' matched to folder '%s'",
                                value, match)
                mapped.append(match)
        table["id"] = mapped

        if len(unmatched) > 0:
            logger.warning("%d CSV id(s) do not match any participant folder: "
                           "%s", len(unmatched),
                           ", ".join("'%s'" % v for v in unmatched[:20]))

    # -- sanity checks -------------------------------------------------------
    valid_sides = set(cfg.CONTRALATERAL_ROI.keys())
    unexpected = sorted(set(table[cfg.SIDE_COLUMN]) - valid_sides)
    if len(unexpected) > 0:
        logger.warning("Unexpected values in '%s': %s. Only %s are understood.",
                       side_col, unexpected, sorted(valid_sides))

    duplicates = table["id"][table["id"].duplicated(keep=False)]
    if len(duplicates) > 0:
        logger.warning("These ids appear more than once in the CSV: %s. "
                       "The FIRST row for each is used.",
                       sorted(set(duplicates)))
        table = table.drop_duplicates(subset=["id"], keep="first")

    logger.info("Loaded stimulator side for %d participants from %s",
                len(table), csv_path)
    for _, row in table.iterrows():
        logger.info("    %-8s -> stimulator %s -> target ROI %s",
                    row["id"], row[cfg.SIDE_COLUMN],
                    cfg.CONTRALATERAL_ROI.get(row[cfg.SIDE_COLUMN], "UNKNOWN"))

    return table


# Word-like pieces that mean "independent component".
COMPONENT_WORDS = {"ic", "ics", "ica", "icas", "comp", "comps",
                   "component", "components"}

# Substrings that mean "was thrown away".
REJECTION_WORDS = ["exclud", "reject", "bad", "drop", "remov"]


def looks_like_ic_reject_key(key_path):
    """
    Decide whether a JSON key plausibly holds the rejected components.

    The FULL key path is examined, so a nested key such as "ica.exclude"
    is recognised: "ica" supplies the component part and "exclude" the
    rejection part. The path is split into words, and one of them must be
    a component word, which keeps "n_epochs_rejected" from matching.
    """
    key_path = key_path.lower()

    words = []
    current = ""
    for character in key_path:
        if character.isalnum():
            current = current + character
        else:
            if current != "":
                words.append(current)
            current = ""
    if current != "":
        words.append(current)

    mentions_components = False
    for word in words:
        if word in COMPONENT_WORDS:
            mentions_components = True

    mentions_rejection = False
    for marker in REJECTION_WORDS:
        if marker in key_path:
            mentions_rejection = True

    return mentions_components and mentions_rejection


def walk_json(node, path=""):
    """
    Walk a nested JSON structure and yield every (key_path, value) pair.
    Used both to find the IC count and to report what the file contains.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = key if path == "" else "%s.%s" % (path, key)
            yield child_path, value
            for item in walk_json(value, child_path):
                yield item
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child_path = "%s[%d]" % (path, index)
            for item in walk_json(value, child_path):
                yield item


def load_manifest(logger):
    """
    Read the stage 0 manifest into a dictionary keyed by
    (subject, recording), e.g. ("hx_01", "pre_task_rest").

    The manifest is the best source for the rejected IC count and for
    cross-checking block detection, because stage 0 wrote it. Returns an
    empty dictionary if the file is missing or unreadable.
    """
    path = getattr(cfg, "MANIFEST_CSV", None)
    if not path or not os.path.exists(path):
        logger.warning("Stage 0 manifest not found at %s. Falling back to the "
                       "per-session JSON files.", path)
        return {}

    manifest = pd.read_csv(path)
    manifest = manifest.dropna(subset=["subject"])

    table = {}
    for _, row in manifest.iterrows():
        key = (str(row["subject"]).strip(), str(row["recording"]).strip())
        table[key] = row

    logger.info("Loaded stage 0 manifest: %d rows covering %d subjects",
                len(table), manifest["subject"].nunique())
    return table


def load_expected_conditions(logger, known_ids=None):
    """
    Read the intended condition of every rest block from the participant
    info CSV.

    Returns a dictionary keyed by (subject, session) holding a list of
    "ON"/"OFF" in chronological block order, e.g.
        ("hx_01", "pre_task_rest") -> ["OFF", "ON", "OFF", "ON"]

    Returns an empty dictionary if the file or the columns are missing, in
    which case validation is skipped.
    """
    path = getattr(cfg, "PARTICIPANT_INFO_CSV", None) or cfg.CONTRALATERALITY_CSV
    if not path or not os.path.exists(path):
        logger.warning("Participant info CSV not found at %s. Block-order "
                       "validation will be SKIPPED.", path)
        return {}

    table = pd.read_csv(path)
    columns = list(table.columns)

    id_col = find_column(columns,
                         [cfg.ID_COLUMN, "participant id", "participant_id",
                          "participant", "subject id", "subject", "id"],
                         logger, "participant id")
    if id_col is None:
        logger.warning("No participant id column in %s. Block-order "
                       "validation will be SKIPPED.", path)
        return {}

    # Locate the four block columns for each session.
    resolved = {}
    for session, headers in cfg.BLOCK_COLUMNS.items():
        found = []
        for header in headers:
            column = find_column(columns, [header], logger, "block")
            found.append(column)
        if any(column is None for column in found):
            missing = [headers[i] for i, c in enumerate(found) if c is None]
            logger.warning("Could not find these block columns for %s: %s. "
                           "Block-order validation will be SKIPPED for that "
                           "session.\nColumns present: %s",
                           session, missing, columns)
        else:
            resolved[session] = found
            logger.info("Block columns for %s: %s", session,
                        ", ".join("'%s'" % c for c in found))

    if len(resolved) == 0:
        return {}

    expected = {}
    for _, row in table.iterrows():
        subject = str(row[id_col]).strip()
        if known_ids:
            matched = normalise_participant_id(subject, known_ids)
            if matched is not None:
                subject = matched

        for session, block_columns in resolved.items():
            conditions = []
            usable = True
            for column in block_columns:
                value = row[column]
                if pd.isna(value):
                    usable = False
                    break
                key = str(value).strip().lower()
                condition = cfg.CONDITION_VALUES.get(key)
                if condition is None:
                    # tolerate "Stimulation (ON)" and similar
                    for name in cfg.CONDITION_VALUES:
                        if key.startswith(name):
                            condition = cfg.CONDITION_VALUES[name]
                            break
                if condition is None:
                    logger.warning("%s / %s: unrecognised condition value "
                                   "'%s' in column '%s'. Expected one of %s.",
                                   subject, session, value, column,
                                   sorted(set(cfg.CONDITION_VALUES)))
                    usable = False
                    break
                conditions.append(condition)

            if usable and len(conditions) > 0:
                key = (subject, session)
                if key in expected and expected[key] != conditions:
                    logger.warning("%s / %s appears more than once in %s with "
                                   "different block orders (%s vs %s). The "
                                   "FIRST is used.", subject, session, path,
                                   expected[key], conditions)
                elif key not in expected:
                    expected[key] = conditions

    logger.info("Loaded expected block order for %d participant-sessions "
                "from %s", len(expected), path)
    return expected


def align_found_to_expected(found, expected):
    """
    Work out which expected block each found block corresponds to.

    `found` and `expected` are lists of "ON"/"OFF". When the two are the
    same length the mapping is positional. When blocks were rejected
    earlier (too short, unpaired) the lists differ in length, and a
    positional comparison would compare block 2 against block 3 and report
    a false mismatch. In that case every order-preserving way of dropping
    the missing blocks is tried; the alignment is used ONLY if exactly one
    of them agrees completely.

    Returns a list the same length as `found` holding the index into
    `expected` for each found block, or None if no unambiguous alignment
    exists.
    """
    if len(found) == 0 or len(expected) == 0:
        return None

    if len(found) == len(expected):
        return list(range(len(found)))

    if len(found) > len(expected):
        return None

    from itertools import combinations
    matches = []
    for indices in combinations(range(len(expected)), len(found)):
        if all(expected[index] == condition
               for index, condition in zip(indices, found)):
            matches.append(list(indices))

    if len(matches) == 1:
        return matches[0]
    return None


def parse_stage0_log(logger):
    """
    Read the ICLabel class counts out of the stage 0 console log.

    The log lines look like:
        [M2] ...  INFO       [hx_01/post_task_rest] 1 file(s)
        ...
        [M2] ...  INFO         ICLabel: {'brain': 25, 'eye blink': 4, ...}

    The log is written by several worker processes at once, so lines from
    different recordings are interleaved. Each line starts with a worker
    tag such as [M1] or [M2]; tracking the "current recording" SEPARATELY
    PER WORKER is what keeps the counts attached to the right file.

    Returns a dictionary keyed by (subject, recording) holding the class
    counts, e.g. {"brain": 25, "eye blink": 4, "other": 31}.
    """
    path = getattr(cfg, "STAGE0_LOG", None)
    if not path or not os.path.exists(path):
        logger.warning("Stage 0 log not found at %s. The ICLabel brain-count "
                       "check will be SKIPPED.", path)
        return {}

    worker_pattern = re.compile(r"^\[(?P<worker>[A-Za-z0-9]+)\]")
    header_pattern = re.compile(r"\[(?P<subject>hx_\d+)/(?P<recording>[A-Za-z0-9_]+)\]")
    iclabel_pattern = re.compile(r"ICLabel:\s*(?P<payload>\{.*\})\s*$")

    current_by_worker = {}
    counts_by_recording = {}

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            match = worker_pattern.match(line)
            worker = match.group("worker") if match else "_"

            header = header_pattern.search(line)
            if header:
                current_by_worker[worker] = (header.group("subject"),
                                             header.group("recording"))

            iclabel = iclabel_pattern.search(line)
            if iclabel and worker in current_by_worker:
                text = iclabel.group("payload")
                try:
                    # ast.literal_eval only evaluates literals, so this is
                    # safe on text read from a file.
                    counts = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    logger.warning("Could not parse an ICLabel line: %s",
                                   text[:120])
                    continue
                if isinstance(counts, dict):
                    # A later run of stage 0 overwrites an earlier one,
                    # which is what we want from an append-only log.
                    counts_by_recording[current_by_worker[worker]] = counts

    logger.info("Read ICLabel counts for %d recordings from %s",
                len(counts_by_recording), path)
    return counts_by_recording


def manifest_value(row, column):
    """Return a column from a manifest row as a number, or None."""
    if row is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def has_channel_positions(raw):
    """
    Report whether the raw actually carries sensor positions.

    AutoReject interpolates bad channels and therefore needs a montage.
    A file processed by stage 0 without an .elc montage has none, and the
    positions come back as NaN or all-zero.
    """
    montage = raw.get_montage()
    if montage is None:
        return False

    positions = montage.get_positions().get("ch_pos", {})
    if len(positions) == 0:
        return False

    for name in positions:
        coordinates = np.asarray(positions[name], dtype=float)
        if np.all(np.isfinite(coordinates)) and np.any(coordinates != 0):
            return True

    return False


def get_rejected_ic_count(json_path, logger=None, label=""):
    """
    Read a stage 0 JSON file and find how many ICs were rejected.

    The exact key name is not known in advance, so the whole structure is
    searched (including nested dictionaries) for a key that mentions both
    components and rejection. A whole number is taken as the count; a list
    is taken as the excluded indices and its length is used.

    Returns None if nothing matches. In that case the keys that ARE present
    are written to the log, so the config can be corrected.
    """
    if not os.path.exists(json_path):
        if logger is not None:
            logger.warning("%s: stage 0 JSON not found at %s", label, json_path)
        return None

    with open(json_path, "r") as f:
        info = json.load(f)

    # If the user has named the key explicitly in the config, trust that.
    explicit = getattr(cfg, "IC_REJECT_JSON_KEY", None)
    if explicit:
        for key_path, value in walk_json(info):
            if key_path == explicit or key_path.endswith("." + explicit):
                if isinstance(value, list):
                    return len(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)

    # Otherwise search for a plausible key.
    for key_path, value in walk_json(info):
        if not looks_like_ic_reject_key(key_path):
            continue
        if isinstance(value, list):
            if logger is not None:
                logger.info("%s: read the IC count from '%s' (a list of %d "
                            "entries)", label, key_path, len(value))
            return len(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if logger is not None:
                logger.info("%s: read the IC count from '%s' (value %s)",
                            label, key_path, value)
            return int(value)

    # Nothing matched. Report what the file actually holds.
    if logger is not None:
        available = []
        for key_path, value in walk_json(info):
            if isinstance(value, (dict, list)) and not (
                    isinstance(value, list) and len(value) < 200):
                continue
            kind = type(value).__name__
            if isinstance(value, list):
                available.append("%s (list of %d)" % (key_path, len(value)))
            else:
                available.append("%s = %s (%s)" % (key_path, value, kind))
        logger.warning("%s: no IC rejection key recognised in %s. Keys "
                       "present: %s", label, os.path.basename(json_path),
                       "; ".join(available) if available else "(none)")

    return None


def trailing_number(text):
    """
    Return the LAST group of digits in a string, or None.

    "Comment/timulus/s1004" -> "1004"
    "Stimulus/S  2"         -> "2"
    """
    groups = re.findall(r"\d+", str(text))
    if len(groups) == 0:
        return None
    return groups[-1]


def select_descriptions(all_descriptions, condition, logger, label):
    """
    Work out which of the annotation descriptions present in a file belong
    to `condition`.

    First tries an exact, case-insensitive match against the strings in
    ANNOTATION_DESCRIPTIONS. If nothing matches, and the fallback is
    enabled, matches on the last group of digits instead and says so.

    Returns a set of the matching descriptions exactly as they appear in
    the file, so the caller can compare them literally.
    """
    wanted = cfg.ANNOTATION_DESCRIPTIONS[condition]
    wanted_normalised = [str(text).strip().lower() for text in wanted]

    matched = set()
    for description in all_descriptions:
        if str(description).strip().lower() in wanted_normalised:
            matched.add(description)

    if len(matched) > 0:
        logger.info("%s: condition %s matched on the exact text %s",
                    label, condition, sorted(matched))
        return matched

    if not getattr(cfg, "ALLOW_ANNOTATION_CODE_FALLBACK", True):
        logger.error("%s: condition %s, no annotation matched %s and the "
                     "code fallback is disabled.", label, condition, wanted)
        return matched

    code = cfg.ANNOTATION_CODES[condition]
    for description in all_descriptions:
        if trailing_number(description) == code:
            matched.add(description)

    if len(matched) > 0:
        logger.warning("%s: condition %s did not match the configured text "
                       "%s. Fell back to the trailing code %s, which matched "
                       "%s. Update ANNOTATION_DESCRIPTIONS in h1_config.py.",
                       label, condition, wanted, code, sorted(matched))
    else:
        logger.error("%s: condition %s matched nothing, by text %s or by "
                     "code %s.", label, condition, wanted, code)

    return matched


def find_blocks(raw, condition, min_duration, logger, label):
    """
    Find blocks marked by PAIRS of identical annotations.

    The markers carrying `code` are sorted in time and then scanned from
    the start:

        - take the current marker as the block ONSET
        - take the next marker of the same code as the block END
        - if that pair lasts at least `min_duration`, accept it as a block
          and move on to the marker after the end
        - if it is too short, reject the pair and move the onset forward
          by ONE marker, so the rejected end marker gets its own chance to
          open a block

    Moving forward by one after a rejection is what makes the scan tolerant
    of a spurious extra marker: a stray marker produces one short, rejected
    pair rather than knocking every later pairing out of step.

    Returns a list of (start_time, end_time) tuples.
    """
    # MNE stores annotation onsets in ABSOLUTE recording time, while
    # raw.crop() and raw.times count from the first sample of the data.
    # On a file with a non-zero first_samp the two differ by raw.first_time,
    # so every onset has to be shifted before it can be used to crop.
    first_time = float(raw.first_time)

    # Every distinct annotation text present in this recording.
    all_descriptions = sorted({str(a["description"]) for a in raw.annotations})

    wanted_descriptions = select_descriptions(all_descriptions, condition,
                                              logger, label)
    code = cfg.ANNOTATION_CODES[condition]

    # Keep the description alongside each onset. The two markers of a pair
    # must carry EXACTLY the same text, so the text has to survive to the
    # pairing step rather than being discarded here.
    markers = []
    for annot in raw.annotations:
        description = str(annot["description"])
        if description in wanted_descriptions:
            markers.append((float(annot["onset"]) - first_time, description))

    markers.sort(key=lambda item: item[0])
    onsets = [onset for onset, _ in markers]

    if len(onsets) == 0:
        logger.warning("%s: no annotations found for condition %s. The file "
                       "contains these descriptions: %s",
                       label, condition,
                       ", ".join("'%s'" % d for d in all_descriptions[:40]))
        return []

    logger.info("%s: %d annotations found for code %s "
                "(first_time = %.3f s, data ends at %.3f s, "
                "markers span %.3f s to %.3f s after correction)",
                label, len(onsets), code, first_time, raw.times[-1],
                onsets[0], onsets[-1])

    # If the corrected onsets still fall outside the data, something is
    # wrong with the file rather than with the arithmetic. Say so clearly.
    out_of_range = [t for t in onsets if t < 0 or t > raw.times[-1]]
    if len(out_of_range) > 0:
        logger.error("%s: code %s, %d marker(s) fall outside the recording "
                     "even after the first_time correction (e.g. %.3f s, "
                     "data is 0 to %.3f s). These markers are ignored.",
                     label, code, len(out_of_range), out_of_range[0],
                     raw.times[-1])
        markers = [item for item in markers if 0 <= item[0] <= raw.times[-1]]
        onsets = [onset for onset, _ in markers]
        if len(markers) == 0:
            return []

    blocks = []
    index = 0
    n_rejected = 0
    n_mismatched = 0

    while index < len(markers) - 1:
        start, start_text = markers[index]
        end, end_text = markers[index + 1]
        duration = end - start

        # RULE 1: the closing marker must be EXACTLY the same text as the
        # opening one. Anything else is not a pair.
        if end_text != start_text:
            n_mismatched = n_mismatched + 1
            logger.warning("%s: marker at %.2f s ('%s') is followed by a "
                           "DIFFERENT text at %.2f s ('%s'). Not a pair; "
                           "the opening marker is discarded.",
                           label, start, start_text, end, end_text)
            index = index + 1
            continue

        # RULE 2: the pair must span at least min_duration.
        if duration < min_duration:
            n_rejected = n_rejected + 1
            logger.warning("%s: code %s, pair at %.2f s to %.2f s REJECTED, "
                           "only %.2f s (minimum is %.1f s)",
                           label, code, start, end, duration, min_duration)
            # Only the onset is discarded. The end marker is tried again as
            # the onset of the next candidate block.
            index = index + 1
            continue

        blocks.append((start, end))
        logger.info("%s: code %s, block %d ACCEPTED, %.2f s "
                    "(%.2f s to %.2f s, text '%s')",
                    label, code, len(blocks), duration, start, end, start_text)
        # Both markers belong to this block, so skip past the end marker.
        index = index + 2

    # If the scan ends on a marker with nothing after it, say so.
    if index == len(markers) - 1:
        logger.warning("%s: code %s, the marker at %.2f s has no partner "
                       "after it and is unused",
                       label, code, markers[index][0])

    if n_mismatched > 0:
        logger.error("%s: code %s, %d marker(s) were followed by a different "
                     "annotation text. Check ANNOTATION_DESCRIPTIONS.",
                     label, code, n_mismatched)

    logger.info("%s: code %s, %d block(s) accepted, %d pair(s) rejected",
                label, code, len(blocks), n_rejected)

    # The protocol expects two blocks per condition per session. Anything
    # else is worth a look, so it is flagged rather than silently accepted.
    if len(blocks) != 2:
        logger.warning("%s: code %s, expected 2 blocks but accepted %d. "
                       "Check the marker stream for this recording.",
                       label, code, len(blocks))

    return blocks


def validate_block_order(blocks_by_condition, expected, logger, label):
    """
    Check each detected block against the condition the CSV says it should
    have been, and drop the ones that disagree.

    Blocks from both conditions are put back into chronological order
    first, because "block 1" in the CSV means the first block in time, not
    the first block of a particular condition.

    Returns (filtered_blocks_by_condition, report) where report is a list
    of dictionaries, one per detected block.
    """
    ordered = []
    for condition in ["ON", "OFF"]:
        for start, end in blocks_by_condition.get(condition, []):
            ordered.append((start, end, condition))
    ordered.sort(key=lambda item: item[0])

    found = [condition for _, _, condition in ordered]

    if expected is None:
        logger.warning("%s: no expected block order for this session, so the "
                       "blocks are NOT validated.", label)
        return blocks_by_condition, [
            {"block_index": i + 1, "start": s, "end": e, "found": c,
             "expected": "", "verdict": "not_validated"}
            for i, (s, e, c) in enumerate(ordered)]

    logger.info("%s: expected block order from the CSV : %s",
                label, " ".join(expected))
    logger.info("%s: detected block order from the EEG : %s",
                label, " ".join(found))

    alignment = align_found_to_expected(found, expected)

    if alignment is None:
        logger.error("%s: cannot line the %d detected block(s) up against the "
                     "%d expected block(s) unambiguously. The blocks are NOT "
                     "validated for this session, and are all kept. Check "
                     "this recording by hand.",
                     label, len(found), len(expected))
        return blocks_by_condition, [
            {"block_index": i + 1, "start": s, "end": e, "found": c,
             "expected": "", "verdict": "alignment_failed"}
            for i, (s, e, c) in enumerate(ordered)]

    kept = {"ON": [], "OFF": []}
    report = []

    for position, (start, end, condition) in enumerate(ordered):
        expected_index = alignment[position]
        expected_condition = expected[expected_index]
        agrees = (condition == expected_condition)

        if agrees:
            kept[condition].append((start, end))
            logger.info("%s: block %d (%.2f-%.2f s) is %s, CSV block %d says "
                        "%s. MATCH.", label, position + 1, start, end,
                        condition, expected_index + 1, expected_condition)
            verdict = "match"
        else:
            logger.error("%s: block %d (%.2f-%.2f s) is marked %s in the EEG "
                         "but CSV block %d says %s. MISMATCH -- block %s.",
                         label, position + 1, start, end, condition,
                         expected_index + 1, expected_condition,
                         "DROPPED" if cfg.DROP_BLOCKS_FAILING_VALIDATION
                         else "KEPT (dropping is disabled)")
            verdict = "mismatch_dropped" if cfg.DROP_BLOCKS_FAILING_VALIDATION \
                else "mismatch_kept"
            if not cfg.DROP_BLOCKS_FAILING_VALIDATION:
                kept[condition].append((start, end))

        report.append({"block_index": position + 1, "start": start,
                       "end": end, "found": condition,
                       "expected": expected_condition, "verdict": verdict})

    return kept, report


def epochs_from_blocks(raw, blocks, condition, logger, label):
    """
    Cut fixed-length epochs INSIDE each block, then pool them.

    Epoching happens block by block so that no epoch straddles the gap
    between two blocks that were recorded minutes apart. The block number
    is stored in the epochs metadata.
    """
    epochs_per_block = []

    for block_number, (start, end) in enumerate(blocks, start=1):
        # Keep the window inside the data. Clamping the end is safe, but a
        # start that is already past the end of the recording means the
        # block is unusable, so skip it loudly rather than crop backwards.
        data_end = raw.times[-1]

        if start >= data_end:
            logger.error("%s: block %d (%s) starts at %.3f s but the "
                         "recording ends at %.3f s. Block skipped.",
                         label, block_number, condition, start, data_end)
            continue

        if end > data_end:
            logger.warning("%s: block %d (%s) ends at %.3f s, past the end "
                           "of the recording (%.3f s). Trimmed.",
                           label, block_number, condition, end, data_end)
            end = data_end

        # Trim the block down to a whole number of epochs. MNE would drop
        # the trailing partial epoch anyway, but doing it explicitly means
        # the log records the duration actually used.
        raw_duration = end - start
        n_whole = int(raw_duration // cfg.EPOCH_LENGTH)
        if n_whole == 0:
            logger.warning("%s: block %d (%s) is only %.2f s, shorter than "
                           "one %.0f s epoch. Skipped.",
                           label, block_number, condition, raw_duration,
                           cfg.EPOCH_LENGTH)
            continue
        usable = n_whole * cfg.EPOCH_LENGTH
        if abs(raw_duration - usable) > 1e-6:
            logger.info("%s: block %d (%s) is %.2f s -> cropped to %.0f s "
                        "(%d epochs), %.2f s of remainder dropped",
                        label, block_number, condition, raw_duration,
                        usable, n_whole, raw_duration - usable)
        end = start + usable

        block_raw = raw.copy().crop(tmin=start, tmax=end)

        block_epochs = mne.make_fixed_length_epochs(
            block_raw,
            duration=cfg.EPOCH_LENGTH,
            overlap=cfg.EPOCH_OVERLAP,
            preload=True,
            verbose=False,
        )

        if len(block_epochs) == 0:
            logger.warning("%s: block %d (%s) produced no epochs",
                           label, block_number, condition)
            continue

        metadata = pd.DataFrame({
            "condition": [condition] * len(block_epochs),
            "block": [block_number] * len(block_epochs),
            "epoch_in_block": list(range(1, len(block_epochs) + 1)),
        })
        block_epochs.metadata = metadata
        epochs_per_block.append(block_epochs)

        logger.info("%s: block %d (%s) -> %d epochs",
                    label, block_number, condition, len(block_epochs))

    if len(epochs_per_block) == 0:
        return None

    if len(epochs_per_block) == 1:
        return epochs_per_block[0]

    return mne.concatenate_epochs(epochs_per_block, verbose=False)


def run_autoreject(epochs, logger, label):
    """
    Fit AutoReject once on the supplied epochs and return the cleaned
    epochs plus the rejection log.

    This is called with ON and OFF epochs TOGETHER, so both conditions
    face identical thresholds.
    """
    ar = AutoReject(random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS,
                    verbose=False)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    logger.info("%s: AutoReject kept %d of %d epochs",
                label, len(epochs_clean), len(epochs))
    return epochs_clean, reject_log


def compute_roi_spectra(epochs, roi_name):
    """
    Compute a Welch PSD and average it across the channels of one ROI.

    Returns (freqs, spectra) where spectra has shape (n_epochs, n_freqs).
    Returns (None, None) if the ROI channels are missing from the data.
    """
    wanted = cfg.ROI_CHANNELS[roi_name]
    available = [ch for ch in wanted if ch in epochs.ch_names]
    if len(available) == 0:
        return None, None

    sfreq = epochs.info["sfreq"]

    n_per_seg = int(cfg.WELCH_WINDOW_SEC * sfreq)
    n_overlap = int(cfg.WELCH_OVERLAP_SEC * sfreq)

    psd = epochs.compute_psd(
        method="welch",
        fmin=cfg.PSD_FMIN,
        fmax=cfg.PSD_FMAX,
        picks=available,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        n_jobs=cfg.N_JOBS,
        verbose=False,
    )

    freqs = psd.freqs
    data = psd.get_data()          # (n_epochs, n_channels, n_freqs)
    spectra = data.mean(axis=1)    # average across the ROI channels
    return freqs, spectra


def fit_one_spectrum(freqs, spectrum):
    """
    Fit FOOOF to a single power spectrum.

    Returns a dictionary with the fit error, the r-squared, and the area
    under the flattened (aperiodic-subtracted) spectrum in the 2-3 Hz band.
    Returns None if the fit fails.
    """
    fm = FOOOF(
        peak_width_limits=cfg.FOOOF_SETTINGS["peak_width_limits"],
        max_n_peaks=cfg.FOOOF_SETTINGS["max_n_peaks"],
        min_peak_height=cfg.FOOOF_SETTINGS["min_peak_height"],
        peak_threshold=cfg.FOOOF_SETTINGS["peak_threshold"],
        aperiodic_mode=cfg.FOOOF_SETTINGS["aperiodic_mode"],
        verbose=False,
    )

    try:
        fm.fit(freqs, spectrum, cfg.FOOOF_FIT_RANGE)
    except Exception:
        return None

    if not np.isfinite(fm.error_):
        return None

    # The flattened spectrum: the fitted aperiodic component subtracted
    # from the observed spectrum. Both are in log10 power.
    fit_freqs = fm.freqs
    if hasattr(fm, "_spectrum_flat"):
        flattened = fm._spectrum_flat
    else:
        flattened = fm.power_spectrum - fm._ap_fit

    # Area under the flattened spectrum inside the band of interest.
    low, high = cfg.AUC_BAND
    in_band = (fit_freqs >= low) & (fit_freqs <= high)
    if in_band.sum() < 2:
        return None
    auc = area_under_curve(flattened[in_band], fit_freqs[in_band])

    return {
        "auc_2_3": float(auc),
        "fit_error": float(fm.error_),
        "r_squared": float(fm.r_squared_),
        "knee": float(fm.aperiodic_params_[1]) if len(fm.aperiodic_params_) == 3 else np.nan,
        "exponent": float(fm.aperiodic_params_[-1]),
    }


def get_observed_spectrum(fm, freqs=None, spectrum=None):
    """
    Return the observed spectrum, in log10, on the model's frequency axis.

    FOOOFGroup does not always hand its stored data back to the single
    models produced by get_fooof(), so fm.power_spectrum can be None. When
    that happens the spectrum we passed in is used instead. Relying on the
    FOOOF internals alone silently loses every epoch.
    """
    if getattr(fm, "power_spectrum", None) is not None:
        return np.asarray(fm.power_spectrum, dtype=float)

    if freqs is None or spectrum is None:
        return None

    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    in_range = (freqs >= fm.freqs[0]) & (freqs <= fm.freqs[-1])
    observed = np.log10(spectrum[in_range])

    if len(observed) != len(fm.freqs):
        return None
    return observed


def summarise_fooof_model(fm, freqs=None, spectrum=None):
    """
    Pull the numbers we need out of a fitted FOOOF model.
    Returns None if the model did not fit.

    `freqs` and `spectrum` are the data the model was fitted to. They are
    used as a fallback when the model object does not carry its own copy.
    """
    if fm is None or not np.isfinite(fm.error_):
        return None

    fit_freqs = np.asarray(fm.freqs, dtype=float)

    aperiodic_fit = getattr(fm, "_ap_fit", None)
    if aperiodic_fit is None:
        return None
    aperiodic_fit = np.asarray(aperiodic_fit, dtype=float)

    observed = get_observed_spectrum(fm, freqs, spectrum)

    # NOTE: FOOOF sets _spectrum_flat to None on a fresh object, so
    # hasattr() is True even when there is nothing there. Test the VALUE.
    flattened = getattr(fm, "_spectrum_flat", None)
    if flattened is None:
        if observed is None:
            return None
        flattened = observed - aperiodic_fit
    flattened = np.asarray(flattened, dtype=float)

    if observed is None:
        observed = flattened + aperiodic_fit

    low, high = cfg.AUC_BAND
    in_band = (fit_freqs >= low) & (fit_freqs <= high)
    if in_band.sum() < 2:
        return None

    aperiodic = fm.aperiodic_params_

    # The full FOOOF model (aperiodic + fitted peaks). Kept for the group
    # average figure, not written to the CSV.
    modelled = getattr(fm, "fooofed_spectrum_", None)
    if modelled is None:
        modelled = observed
    modelled = np.asarray(modelled, dtype=float)

    return {
        "auc_2_3": float(area_under_curve(flattened[in_band],
                                          fit_freqs[in_band])),
        "fit_error": float(fm.error_),
        "r_squared": float(fm.r_squared_),
        "knee": float(aperiodic[1]) if len(aperiodic) == 3 else np.nan,
        "exponent": float(aperiodic[-1]),
        "_curves": {
            "freqs": fit_freqs,
            "observed": observed,
            "modelled": modelled,
            "aperiodic": aperiodic_fit,
            "flattened": flattened,
        },
    }


def fit_all_spectra(freqs, spectra, logger, label):
    """
    Fit FOOOF to every epoch spectrum at once using FOOOFGroup.

    FOOOFGroup runs the fits in parallel, which is where most of the time
    in this pipeline goes. Returns a list with one dictionary per spectrum
    (or None where the fit failed), in the same order as the input.
    """
    fg = FOOOFGroup(
        peak_width_limits=cfg.FOOOF_SETTINGS["peak_width_limits"],
        max_n_peaks=cfg.FOOOF_SETTINGS["max_n_peaks"],
        min_peak_height=cfg.FOOOF_SETTINGS["min_peak_height"],
        peak_threshold=cfg.FOOOF_SETTINGS["peak_threshold"],
        aperiodic_mode=cfg.FOOOF_SETTINGS["aperiodic_mode"],
        verbose=False,
    )

    try:
        fg.fit(freqs, spectra, cfg.FOOOF_FIT_RANGE, n_jobs=cfg.N_JOBS)
    except Exception as error:
        logger.error("%s: the FOOOF group fit failed (%s). Falling back to "
                     "fitting each epoch one at a time.", label, error)
        return [fit_one_spectrum(freqs, spectra[i, :])
                for i in range(spectra.shape[0])]

    results = []
    first_error = None
    for index in range(spectra.shape[0]):
        try:
            fm = fg.get_fooof(index, regenerate=True)
            results.append(summarise_fooof_model(fm, freqs, spectra[index, :]))
        except Exception as error:
            if first_error is None:
                first_error = error
            results.append(None)

    n_ok = sum(1 for r in results if r is not None)

    # If the group path produced nothing usable, fall back to fitting each
    # epoch individually rather than silently returning a list of Nones.
    if n_ok == 0 and spectra.shape[0] > 0:
        logger.error("%s: FOOOFGroup returned no usable models (first error: "
                     "%s). Refitting each epoch individually.",
                     label, first_error)
        results = [fit_one_spectrum(freqs, spectra[i, :])
                   for i in range(spectra.shape[0])]
        n_ok = sum(1 for r in results if r is not None)
        if n_ok == 0:
            logger.error("%s: individual fitting also failed for all %d "
                         "epochs.", label, spectra.shape[0])
    elif first_error is not None:
        logger.warning("%s: %d of %d FOOOF models could not be read "
                       "(first error: %s)", label,
                       spectra.shape[0] - n_ok, spectra.shape[0], first_error)

    return results


def plot_group_spectra(spectrum_store, output_dir, logger):
    """
    Average power spectrum across every retained epoch of every included
    participant, from the contralateral ROI.

    Three panels:
        left   the observed spectrum
        middle the observed spectrum with the FOOOF model on top
        right  the flattened spectrum, with the 2-3 Hz band shaded

    Colour carries the condition and line style carries the session, so
    the four series are told apart by two cues rather than colour alone.
    """
    # matplotlib is imported here so the extraction itself does not depend
    # on it being installed.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed, so the group spectrum "
                       "figure is skipped.")
        return None

    if len(spectrum_store) == 0:
        logger.error("No spectra were collected, so there is no group figure "
                     "to draw. Every FOOOF fit failed, or no participant "
                     "survived the exclusion rules.")
        return None

    collected = sum(len(store["observed"]) for store in spectrum_store.values())
    logger.info("Group figure will average %d epoch spectra across %d cells",
                collected, len(spectrum_store))
    if collected == 0:
        logger.error("Every cell is empty, so there is no group figure to draw.")
        return None

    surface = "#fcfcfb"
    ink = "#1a1a19"
    ink_soft = "#6b6659"
    grid = "#e3dfd7"
    colour = {"ON": "#2a78d6", "OFF": "#eb6834"}
    style = {"pre_task_rest": "-", "post_task_rest": "--"}

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=surface)

    for axis in axes:
        axis.set_facecolor(surface)
        axis.grid(True, color=grid, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(grid)
        axis.spines["bottom"].set_color(grid)
        axis.tick_params(colors=ink_soft)
        axis.set_xscale("log")
        axis.set_xlabel("frequency (Hz)", color=ink)

    low, high = cfg.AUC_BAND
    total_epochs = 0

    for session in ["pre_task_rest", "post_task_rest"]:
        for condition in ["ON", "OFF"]:
            store = spectrum_store.get((session, condition))
            if store is None or len(store["observed"]) == 0:
                continue

            freqs = store["freqs"]
            observed = np.mean(np.vstack(store["observed"]), axis=0)
            modelled = np.mean(np.vstack(store["modelled"]), axis=0)
            flattened = np.mean(np.vstack(store["flattened"]), axis=0)
            n_epochs = len(store["observed"])
            total_epochs = total_epochs + n_epochs

            label = "%s %s (n=%d)" % (session.replace("_task_rest", ""),
                                      condition, n_epochs)
            line_style = style[session]
            line_colour = colour[condition]

            axes[0].plot(freqs, observed, line_style, color=line_colour,
                         linewidth=2.0, label=label)
            axes[1].plot(freqs, observed, line_style, color=line_colour,
                         linewidth=1.2, alpha=0.45)
            axes[1].plot(freqs, modelled, line_style, color=line_colour,
                         linewidth=2.0, label=label)
            axes[2].plot(freqs, flattened, line_style, color=line_colour,
                         linewidth=2.0, label=label)

            in_band = (freqs >= low) & (freqs <= high)
            axes[2].fill_between(freqs[in_band], 0, flattened[in_band],
                                 color=line_colour, alpha=0.18)

    axes[0].set_ylabel("log10 power", color=ink)
    axes[0].set_title("average power spectrum", loc="left", fontsize=11,
                      color=ink)
    axes[1].set_title("FOOOF model over the observed spectrum", loc="left",
                      fontsize=11, color=ink)
    axes[2].set_title("flattened (aperiodic removed)", loc="left",
                      fontsize=11, color=ink)
    axes[2].axhline(0, color=grid, linewidth=1)
    axes[2].axvspan(low, high, color=ink_soft, alpha=0.06)

    for axis in axes:
        axis.legend(frameon=False, fontsize=8.5, labelcolor=ink)

    figure.suptitle("Group average across %d epochs, contralateral ROI  "
                    "(solid = pre, dashed = post)" % total_epochs,
                    fontsize=12, color=ink, y=0.99)
    figure.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(output_dir, "h1_group_spectra.png")
    figure.savefig(path, dpi=140, facecolor=surface)
    plt.close(figure)

    logger.info("Group spectrum figure written to %s (%d epochs)",
                path, total_epochs)
    return path


def save_group_spectra_csv(spectrum_store, output_dir, logger):
    """Write the averaged curves so they can be re-plotted without re-running."""
    rows = []
    for (session, condition), store in spectrum_store.items():
        if len(store["observed"]) == 0:
            continue
        freqs = store["freqs"]
        observed = np.mean(np.vstack(store["observed"]), axis=0)
        modelled = np.mean(np.vstack(store["modelled"]), axis=0)
        aperiodic = np.mean(np.vstack(store["aperiodic"]), axis=0)
        flattened = np.mean(np.vstack(store["flattened"]), axis=0)
        for index in range(len(freqs)):
            rows.append({
                "session": session,
                "condition": condition,
                "n_epochs": len(store["observed"]),
                "freq": freqs[index],
                "observed_log10": observed[index],
                "fooof_model_log10": modelled[index],
                "aperiodic_log10": aperiodic[index],
                "flattened_log10": flattened[index],
            })

    if len(rows) == 0:
        return None

    path = os.path.join(output_dir, "h1_group_spectra.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Group spectrum values written to %s", path)
    return path


# ===========================================================================
# Main routine
# ===========================================================================

def process_participant(participant_id, side, logger, manifest,
                        iclabel_counts, spectrum_store,
                        expected_conditions, validation_rows):
    """
    Run the whole extraction for one participant.

    Returns three things:
        epoch_rows    list of dictionaries, one per epoch per ROI
        qc_rows       list of dictionaries, one per session per condition
        exclusion     None, or a dictionary describing why the participant
                      was excluded
    """
    folder = os.path.join(cfg.DERIVATIVES_DIR, participant_id)
    epoch_rows = []
    qc_rows = []

    # -- target ROI ---------------------------------------------------------
    if side not in cfg.CONTRALATERAL_ROI:
        return [], [], {"id": participant_id, "reason": "unknown_stimulator_side",
                        "detail": str(side)}
    target_roi = cfg.CONTRALATERAL_ROI[side]
    logger.info("%s: stimulator on %s hand -> target ROI is %s",
                participant_id, side, target_roi)

    # -- stage 0 ICA check --------------------------------------------------
    max_ics = int(cfg.MAX_IC_REJECT_FRACTION * cfg.TOTAL_ICS)
    for session in ["pre_task_rest", "post_task_rest"]:
        session_label = "%s / %s" % (participant_id, session)
        manifest_row = manifest.get((participant_id, session))

        # The manifest is the preferred source; the JSON is the fallback.
        n_rejected = manifest_value(manifest_row, "n_ic_rejected")
        if n_rejected is not None:
            n_rejected = int(n_rejected)
            logger.info("%s: %d ICs rejected (from the stage 0 manifest)",
                        session_label, n_rejected)
        else:
            json_path = os.path.join(
                folder, "%s_%s_stage0.json" % (participant_id, session))
            n_rejected = get_rejected_ic_count(json_path, logger, session_label)

        if n_rejected is None:
            logger.warning("%s: the noisy-ICA check is SKIPPED for this "
                           "session. Set IC_REJECT_JSON_KEY in h1_config.py "
                           "to the correct key name to enable it.",
                           session_label)
        elif n_rejected >= max_ics:
            return [], [], {"id": participant_id, "reason": "noisy_ica",
                            "detail": "%s: %d of %d ICs rejected"
                                      % (session, n_rejected, cfg.TOTAL_ICS)}

        # -- ICLabel brain-component check ----------------------------------
        counts = iclabel_counts.get((participant_id, session))
        if counts is None:
            logger.warning("%s: no ICLabel counts found in the stage 0 log. "
                           "The brain-component check is SKIPPED.",
                           session_label)
        else:
            n_brain = int(counts.get("brain", 0))
            summary = ", ".join("%s %d" % (name, counts[name])
                                for name in sorted(counts))
            logger.info("%s: ICLabel -> %s  (brain = %d)",
                        session_label, summary, n_brain)
            if n_brain < cfg.MIN_BRAIN_ICS:
                return [], [], {
                    "id": participant_id,
                    "reason": "too_few_brain_ics",
                    "detail": "%s: only %d brain components (minimum is %d)"
                              % (session, n_brain, cfg.MIN_BRAIN_ICS),
                }

    # -- per session --------------------------------------------------------
    clean_epochs_by_session = {}

    for session in ["pre_task_rest", "post_task_rest"]:
        label = "%s / %s" % (participant_id, session)
        fif_path = os.path.join(
            folder, "%s_%s_clean-raw.fif" % (participant_id, session))

        if not os.path.exists(fif_path):
            return [], [], {"id": participant_id, "reason": "missing_file",
                            "detail": fif_path}

        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        logger.info("%s: loaded, %.1f s of data, %d channels",
                    label, raw.times[-1], len(raw.ch_names))

        # -- sensor positions ------------------------------------------------
        # Stage 0 logged "montage: NOT FOUND" for some recordings. Without
        # positions AutoReject cannot interpolate, so the file is unusable.
        if not has_channel_positions(raw):
            message = ("no sensor positions in %s (stage 0 found no montage). "
                       "AutoReject needs a montage." % os.path.basename(fif_path))
            if cfg.REQUIRE_CHANNEL_POSITIONS:
                return [], [], {"id": participant_id,
                                "reason": "no_channel_positions",
                                "detail": message}
            logger.warning("%s: %s Continuing anyway because "
                           "REQUIRE_CHANNEL_POSITIONS is False.", label, message)

        # -- ROI channels ----------------------------------------------------
        # If stage 0 could not rename channels to the montage, the ROI names
        # may simply not exist in this file.
        for roi_name in ["left", "right"]:
            wanted = cfg.ROI_CHANNELS[roi_name]
            missing = [ch for ch in wanted if ch not in raw.ch_names]
            if len(missing) == len(wanted):
                return [], [], {"id": participant_id,
                                "reason": "roi_channels_missing",
                                "detail": "%s: none of the %s ROI channels "
                                          "(%s) exist in this file"
                                          % (session, roi_name,
                                             ", ".join(wanted))}
            if len(missing) > 0:
                logger.warning("%s: %s ROI is missing %d of %d channels (%s). "
                               "The average uses the %d that are present.",
                               label, roi_name, len(missing), len(wanted),
                               ", ".join(missing), len(wanted) - len(missing))

        # Find blocks and build epochs for each condition.
        epochs_by_condition = {}
        blocks_by_condition = {}
        for condition in ["ON", "OFF"]:
            code = cfg.ANNOTATION_CODES[condition]
            blocks = find_blocks(raw, condition, cfg.MIN_BLOCK_DURATION,
                                 logger, label)

            blocks_by_condition[condition] = blocks

            # Cross-check against what stage 0 counted for this recording.
            manifest_row = manifest.get((participant_id, session))
            expected_markers = manifest_value(manifest_row,
                                              "n_ann_%s" % code)
            if expected_markers is not None:
                logger.info("%s: stage 0 counted %d annotations with code %s; "
                            "this script accepted %d block(s)",
                            label, int(expected_markers), code, len(blocks))

            if len(blocks) == 0:
                return [], [], {"id": participant_id, "reason": "no_valid_blocks",
                                "detail": "%s, condition %s" % (session, condition)}

        # -- validate the block order against the participant info CSV ------
        # This happens AFTER both conditions are found, because "block 1" in
        # the CSV means the first block in time, not the first ON block.
        blocks_by_condition, block_report = validate_block_order(
            blocks_by_condition,
            expected_conditions.get((participant_id, session)),
            logger, label)

        for entry in block_report:
            entry.update({"id": participant_id, "session": session})
        validation_rows.extend(block_report)

        for condition in ["ON", "OFF"]:
            if len(blocks_by_condition.get(condition, [])) == 0:
                return [], [], {
                    "id": participant_id,
                    "reason": "no_blocks_after_validation",
                    "detail": "%s, condition %s: every block disagreed with "
                              "the participant info CSV" % (session, condition),
                }

        # -- build the epochs ------------------------------------------------
        for condition in ["ON", "OFF"]:
            condition_epochs = epochs_from_blocks(
                raw, blocks_by_condition[condition], condition, logger, label)
            if condition_epochs is None:
                return [], [], {"id": participant_id, "reason": "no_epochs",
                                "detail": "%s, condition %s" % (session, condition)}
            epochs_by_condition[condition] = condition_epochs

        # Put ON and OFF together so AutoReject is fitted once per session.
        all_epochs = mne.concatenate_epochs(
            [epochs_by_condition["ON"], epochs_by_condition["OFF"]],
            verbose=False)

        n_total_by_condition = {
            "ON": len(epochs_by_condition["ON"]),
            "OFF": len(epochs_by_condition["OFF"]),
        }

        epochs_clean, reject_log = run_autoreject(all_epochs, logger, label)

        # Count how many epochs AutoReject dropped in each condition.
        conditions_before = all_epochs.metadata["condition"].values
        bad_epochs = np.array(reject_log.bad_epochs)

        n_dropped_by_condition = {}
        for condition in ["ON", "OFF"]:
            is_this_condition = (conditions_before == condition)
            n_dropped_by_condition[condition] = int(
                np.sum(bad_epochs & is_this_condition))

        # The >50% rule.
        for condition in ["ON", "OFF"]:
            n_total = n_total_by_condition[condition]
            n_dropped = n_dropped_by_condition[condition]
            loss = n_dropped / n_total if n_total > 0 else 1.0

            logger.info("%s / %s: %d epochs, %d dropped by AutoReject (%.0f%%)",
                        label, condition, n_total, n_dropped, 100 * loss)

            if loss > cfg.MAX_EPOCH_LOSS:
                return [], [], {
                    "id": participant_id,
                    "reason": "ar_epoch_loss",
                    "detail": "%s / %s: %d of %d epochs dropped (%.0f%%)"
                              % (session, condition, n_dropped, n_total, 100 * loss),
                }

            blocks = blocks_by_condition[condition]
            block_durations = ["%.1f" % (end - start) for start, end in blocks]

            qc_rows.append({
                "id": participant_id,
                "session": session,
                "condition": condition,
                "n_blocks_used": len(blocks),
                "block_durations_s": "; ".join(block_durations),
                "n_epochs_total": n_total,
                "n_dropped_autoreject": n_dropped,
                "n_retained_autoreject": n_total - n_dropped,
            })

        clean_epochs_by_session[session] = epochs_clean

    # -- spectra and FOOOF --------------------------------------------------
    for session in ["pre_task_rest", "post_task_rest"]:
        epochs_clean = clean_epochs_by_session[session]
        metadata = epochs_clean.metadata.reset_index(drop=True)

        for roi_name in ["left", "right"]:
            freqs, spectra = compute_roi_spectra(epochs_clean, roi_name)
            if freqs is None:
                logger.warning("%s / %s: ROI '%s' has no usable channels",
                               participant_id, session, roi_name)
                continue

            fit_results = fit_all_spectra(freqs, spectra, logger,
                                          "%s / %s / %s" % (participant_id,
                                                            session, roi_name))

            for i in range(spectra.shape[0]):
                result = fit_results[i]

                row = {
                    "id": participant_id,
                    "session": session,
                    "condition": metadata.loc[i, "condition"],
                    "block": metadata.loc[i, "block"],
                    "epoch_in_block": metadata.loc[i, "epoch_in_block"],
                    "roi": roi_name,
                    "target_roi": target_roi,
                    "side_of_stimulator": side,
                }

                if result is None:
                    row.update({"auc_2_3": np.nan, "fit_error": np.nan,
                                "r_squared": np.nan, "knee": np.nan,
                                "exponent": np.nan, "fooof_failed": True})
                else:
                    curves = result.pop("_curves", None)
                    row.update(result)
                    row["fooof_failed"] = False

                    # Keep the curves for the group-average figure, but only
                    # for the ROI this participant actually contributes.
                    if curves is not None and roi_name == target_roi:
                        key = (session, row["condition"])
                        store = spectrum_store.setdefault(
                            key, {"freqs": curves["freqs"], "observed": [],
                                  "modelled": [], "aperiodic": [],
                                  "flattened": []})
                        if len(store["freqs"]) == len(curves["freqs"]):
                            store["observed"].append(curves["observed"])
                            store["modelled"].append(curves["modelled"])
                            store["aperiodic"].append(curves["aperiodic"])
                            store["flattened"].append(curves["flattened"])

                epoch_rows.append(row)

        n_failed = sum(1 for r in epoch_rows
                       if r["session"] == session and r["fooof_failed"])
        if n_failed > 0:
            logger.warning("%s / %s: %d FOOOF fits failed outright",
                           participant_id, session, n_failed)

    n_good = sum(1 for r in epoch_rows if not r["fooof_failed"])
    n_curves = sum(len(store["observed"]) for store in spectrum_store.values())
    logger.info("%s: extraction finished, %d epoch-level rows (%d with a "
                "usable FOOOF fit); %d spectra collected for the group "
                "figure so far", participant_id, len(epoch_rows), n_good,
                n_curves)
    if n_good == 0:
        logger.error("%s: NO epoch produced a usable FOOOF fit.",
                     participant_id)
    return epoch_rows, qc_rows, None


def main():
    logger = setup_logging(cfg.OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("H1 extraction started (n_jobs = %s)", cfg.N_JOBS)
    logger.info("PROTOCOL DEVIATION (logged deliberately): section 3.5 assumes "
                "100 s blocks and 10 epochs per condition, and excludes a "
                "participant if more than 5 epochs are removed. The recorded "
                "blocks are ~60 s, so the rule is applied as 'more than %.0f%% "
                "of epochs in any condition'.", 100 * cfg.MAX_EPOCH_LOSS)
    logger.info("=" * 70)

    # The folder names first, so the CSV ids can be matched against them.
    folders = sorted(glob.glob(os.path.join(cfg.DERIVATIVES_DIR, "hx_*")))
    participant_ids = [os.path.basename(f) for f in folders if os.path.isdir(f)]
    logger.info("Found %d participant folders", len(participant_ids))

    side_table = load_side_table(cfg.CONTRALATERALITY_CSV, logger,
                                 known_ids=participant_ids)
    manifest = load_manifest(logger)
    iclabel_counts = parse_stage0_log(logger)
    expected_conditions = load_expected_conditions(logger,
                                                   known_ids=participant_ids)
    all_validation_rows = []

    # Averaged spectra are accumulated here as participants are processed.
    spectrum_store = {}

    all_epoch_rows = []
    all_qc_rows = []
    all_exclusions = []

    for participant_id in participant_ids:
        logger.info("-" * 70)
        logger.info("Processing %s", participant_id)

        match = side_table[side_table["id"] == participant_id]
        if len(match) == 0:
            logger.warning("%s: not found in the contralaterality CSV, skipped",
                           participant_id)
            all_exclusions.append({"id": participant_id,
                                   "reason": "missing_from_side_csv",
                                   "detail": ""})
            continue

        side = match.iloc[0][cfg.SIDE_COLUMN]

        try:
            epoch_rows, qc_rows, exclusion = process_participant(
                participant_id, side, logger, manifest, iclabel_counts,
                spectrum_store, expected_conditions, all_validation_rows)
        except Exception as error:
            logger.error("%s: processing failed with an error: %s",
                         participant_id, error)
            all_exclusions.append({"id": participant_id,
                                   "reason": "processing_error",
                                   "detail": str(error)})
            continue

        if exclusion is not None:
            logger.warning("EXCLUDED %s -- %s (%s)", participant_id,
                           exclusion["reason"], exclusion["detail"])
            all_exclusions.append(exclusion)
            continue

        all_epoch_rows.extend(epoch_rows)
        all_qc_rows.extend(qc_rows)

    # -- write the outputs --------------------------------------------------
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    epoch_table = pd.DataFrame(all_epoch_rows)
    epoch_table.to_csv(os.path.join(cfg.OUTPUT_DIR, "h1_epoch_level.csv"),
                       index=False)

    qc_table = pd.DataFrame(all_qc_rows)
    qc_table.to_csv(os.path.join(cfg.OUTPUT_DIR, "h1_qc_report.csv"), index=False)

    exclusion_table = pd.DataFrame(all_exclusions,
                                   columns=["id", "reason", "detail"])
    exclusion_table.to_csv(os.path.join(cfg.OUTPUT_DIR, "h1_exclusions.csv"),
                           index=False)

    validation_table = pd.DataFrame(
        all_validation_rows,
        columns=["id", "session", "block_index", "start", "end", "found",
                 "expected", "verdict"])
    validation_table.to_csv(
        os.path.join(cfg.OUTPUT_DIR, "h1_block_validation.csv"), index=False)
    if len(validation_table) > 0:
        counts = validation_table["verdict"].value_counts().to_dict()
        logger.info("Block validation across all participants: %s", counts)
        mismatches = validation_table[
            validation_table["verdict"].str.startswith("mismatch")]
        for _, row in mismatches.iterrows():
            logger.error("MISMATCH %s / %s block %s: EEG says %s, CSV says %s",
                         row["id"], row["session"], row["block_index"],
                         row["found"], row["expected"])

    # -- group average spectra, now that every file has been processed ------
    logger.info("-" * 70)
    logger.info("Averaging spectra across all included participants")
    save_group_spectra_csv(spectrum_store, cfg.OUTPUT_DIR, logger)
    plot_group_spectra(spectrum_store, cfg.OUTPUT_DIR, logger)

    logger.info("=" * 70)
    logger.info("Extraction complete")
    logger.info("  participants kept     : %d",
                epoch_table["id"].nunique() if len(epoch_table) > 0 else 0)
    logger.info("  participants excluded : %d", len(all_exclusions))
    if len(all_exclusions) > 0:
        for row in all_exclusions:
            logger.info("      %s -> %s", row["id"], row["reason"])
    logger.info("  epoch-level rows      : %d", len(epoch_table))
    logger.info("Next step: run h1_group.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
