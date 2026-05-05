from pathlib import Path
import mne

def get_processed_epochs(subject_name, condition, subj_save_dir):
    """
    Checks if a preprocessed epochs file already exists for the given condition.
    Returns the loaded mne.Epochs object if it exists, or None if it doesn't.
    """
    expected_fname = subj_save_dir / f"{subject_name}_{condition}-epo.fif"
    
    if expected_fname.exists():
        print(f"  -> Found existing processed data for {condition}. Skipping preprocessing!")
        # Load and return the already processed epochs
        return mne.read_epochs(expected_fname, preload=True)
    
    return None
