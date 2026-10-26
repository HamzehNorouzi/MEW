# Hypothesis 1 pipeline

Resting-state 2–3 Hz change score, STIM-ON vs STIM-OFF.
Implements panels A–D of the approved flowchart.

## Files

| File | What it does |
|---|---|
| `h1_config.py` | Every setting. Edit this, not the scripts. |
| `h1_extract.py` | Panel A. Per participant: blocks → epochs → AutoReject → PSD → FOOOF. |
| `h1_group.py` | Panel B. Group z-score of fit error, change scores, interim samples. |
| `h1_analysis.R` | Panel C. Frequentist + Bayesian tests, and the interim BF record. |

## Order of operations

```bash
python h1_extract.py     # writes h1_epoch_level.csv, h1_qc_report.csv, h1_exclusions.csv
python h1_group.py       # writes h1_participant_level.csv, h1_sequential_input.csv
Rscript h1_analysis.R    # writes h1_main_results.csv, h1_sequential_bf.csv
```

All three append to `derivatives_h1/h1_run.log`.

## Requirements

Python: `mne`, `autoreject`, `fooof` (or `specparam` v1), `pandas`, `numpy`
R: `BayesFactor`

```bash
pip install mne autoreject fooof pandas numpy
```
```r
install.packages("BayesFactor")
```

## Before the first run

1. Put `contralaterality.csv` beside the scripts. It needs a participant id
   column whose values match the folder names exactly (`hx_01`, `hx_02`, …)
   and a column named `Side of stimulator` with values `Right` / `Left`.
   Optionally add an `order` column giving enrolment order — without it the
   sequential analysis assumes participants were collected in number order.
2. Check `DERIVATIVES_DIR` in `h1_config.py` points at your stage-0 output.
3. Check `ANNOTATION_CODES` — currently `1004` = ON, `1002` = OFF.

## Exclusion reasons written to `h1_exclusions.csv`

| Reason | Meaning |
|---|---|
| `noisy_ica` | Stage 0 rejected ≥ 80% of the 60 ICs |
| `no_valid_blocks` | No annotation pair ≥ 59 s for one condition |
| `ar_epoch_loss` | AutoReject dropped > 50% of epochs in one of the four cells |
| `no_epochs_after_specparam` | A cell was emptied by the z > 2 fit screen |
| `missing_file` | A `_clean-raw.fif` was not found |
| `missing_from_side_csv` | Folder present but no row in the contralaterality CSV |
| `processing_error` | Unhandled error; the message is in `detail` and the log |

## Two things worth remembering

**The pipeline is two-pass on purpose.** The FOOOF fit-error z-score is taken
across all participants, so adding participant N+1 changes the group mean and
SD and can change which epochs survive for everyone else. `h1_group.py`
therefore rebuilds each interim sample from scratch rather than appending.
Never hand-edit `h1_sequential_input.csv`.

**The early stopping output is a record, not a decision.** The protocol stops
data collection only when all four hypotheses reach a conclusive Bayes factor.
This pipeline covers H1 alone.
