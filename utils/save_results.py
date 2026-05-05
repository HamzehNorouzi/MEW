import mne
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_preprocessed_results(subject_name, epochs_dict, master_path, 
                              task_tfr=None, task_fig=None, 
                              rest_df=None, rest_fig=None):
    """
    Saves preprocessed epochs, ERD/ERS results, and Specparam results 
    to a designated processed_files folder.
    
    Parameters:
    - subject_name: str, name of the subject
    - epochs_dict: dict, containing the cleaned epochs (e.g., {'task': epochs, 'rest_pre': epochs})
    - master_path: str or Path, the root directory (e.g., '/master/')
    - task_tfr: mne.time_frequency.AverageTFR object
    - task_fig: matplotlib.figure.Figure object
    - rest_df: pandas.DataFrame containing Specparam results
    - rest_fig: matplotlib.figure.Figure object
    """
    
    # 1. Define and create the save directory
    # This creates: /master/preprocessed_files/processed_files/subject_name/
    base_save_dir = Path(master_path) / 'preprocessed_files' / 'processed_files' / subject_name
    
    # parents=True creates all missing intermediate folders
    # exist_ok=True prevents crashes if the folder already exists
    base_save_dir.mkdir(parents=True, exist_ok=True) 
    print(f"\n--- Saving Data for {subject_name} ---")
    
    # ---------------------------------------------------------
    # 2. Save Preprocessed Epochs
    # ---------------------------------------------------------
    for condition, epochs_obj in epochs_dict.items():
        if epochs_obj is not None:
            # MNE requires epochs to end with '-epo.fif'
            epoch_fname = base_save_dir / f"{subject_name}_{condition}-epo.fif"
            epochs_obj.save(epoch_fname, overwrite=True)
            print(f"Saved Epochs: {epoch_fname.name}")

    # ---------------------------------------------------------
    # 3. Save Task Analysis Results (ERD/ERS)
    # ---------------------------------------------------------
    if task_tfr is not None:
        # MNE requires TFR data to end with '-tfr.h5'
        tfr_fname = base_save_dir / f"{subject_name}_task_ERD-tfr.h5"
        task_tfr.save(tfr_fname, overwrite=True)
        print(f"Saved TFR Data: {tfr_fname.name}")
        
    if task_fig is not None:
        task_fig_fname = base_save_dir / f"{subject_name}_task_ERD_plot.png"
        # dpi=300 ensures publication-quality resolution
        task_fig.savefig(task_fig_fname, dpi=300, bbox_inches='tight')
        print(f"Saved Task Plot: {task_fig_fname.name}")

    # ---------------------------------------------------------
    # 4. Save Rest Analysis Results (Specparam)
    # ---------------------------------------------------------
    if rest_df is not None:
        csv_fname = base_save_dir / f"{subject_name}_rest_specparam_results.csv"
        rest_df.to_csv(csv_fname, index=False)
        print(f"Saved Specparam Table: {csv_fname.name}")
        
    if rest_fig is not None:
        rest_fig_fname = base_save_dir / f"{subject_name}_rest_specparam_plot.png"
        rest_fig.savefig(rest_fig_fname, dpi=300, bbox_inches='tight')
        print(f"Saved Rest Plot: {rest_fig_fname.name}")

    print("-" * 40)
