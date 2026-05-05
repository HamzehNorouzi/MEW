from pathlib import Path
import mne

def load_and_concat(file_paths):
    """
    Helper function to load a list of BrainVision files and concatenate them 
    if there is more than one.
    """
    if not file_paths:
        return None
        
    loaded_raws = []
    for f_path in file_paths:
        print(f"  -> Loading: {f_path.name}")
        raw = mne.io.read_raw_brainvision(f_path, preload=True)
        loaded_raws.append(raw)
        
    if len(loaded_raws) > 1:
        # Concatenate if multiple files exist
        return mne.concatenate_raws(loaded_raws)
    else:
        # Just return the single file if only one exists
        return loaded_raws[0]
