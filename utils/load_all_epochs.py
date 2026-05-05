import mne
from pathlib import Path

def load_group_data(condition):
    """
    Loads preprocessed MNE epochs for the specified condition.
    
    Parameters:
    - condition (str): Must be either 'task' or 'rest'.
    
    Returns:
    - dict: A master dictionary containing loaded epochs for all subjects.
    """
    if condition not in ['task', 'rest']:
        raise ValueError("Condition must be exactly 'task' or 'rest'.")

    master_group_data = {}
    master_folder = Path('/home/hamzeh/Documents/MEAOW_project/pilot_data/preprocessed_files')

    # Dictionary to translate expected_file keys to Specparam/Rest keys
    key_translation = {
        'rest_pre_EO_StON': 'EOpen_StON',
        'rest_pre_EO_StOFF': 'EOpen_StOFF',
        'rest_pre_EC_StON': 'EClosed_StON',
        'rest_pre_EC_StOFF': 'EClosed_StOFF',
        'rest_post_EO_StON': 'EOpen_StON',
        'rest_post_EO_StOFF': 'EOpen_StOFF',
        'rest_post_EC_StON': 'EClosed_StON',
        'rest_post_EC_StOFF': 'EClosed_StOFF'
    }

    if not master_folder.exists():
        print(f"[ERROR] Master folder not found at {master_folder}")
        return master_group_data

    for subject_folder in master_folder.iterdir():
        if not subject_folder.is_dir():
            continue
            
        subj_name = subject_folder.name
        
        # Construct subject-specific paths
        subj_save_dir = master_folder / subj_name
        task_folder = subj_save_dir / 'task'
        rest_folder = subj_save_dir / 'rest'

        # Expected files dictionary for this subject
        expected_files = {
            'task': {
                'gait': task_folder / f"{subj_name}_gait-epo.fif",
                'standup': task_folder / f"{subj_name}_standup-epo.fif"
            },
            'rest_post': {
                'rest_post_EO_StON': rest_folder / f"{subj_name}_rest_post_EOpen_StON-epo.fif",
                'rest_post_EO_StOFF': rest_folder / f"{subj_name}_rest_post_EOpen_StOFF-epo.fif",
                'rest_post_EC_StON': rest_folder / f"{subj_name}_rest_post_EClosed_StON-epo.fif",
                'rest_post_EC_StOFF': rest_folder / f"{subj_name}_rest_post_EClosed_StOFF-epo.fif"
            },
            'rest_pre': {
                'rest_pre_EO_StON': rest_folder / f"{subj_name}_rest_pre_EOpen_StON-epo.fif",
                'rest_pre_EO_StOFF': rest_folder / f"{subj_name}_rest_pre_EOpen_StOFF-epo.fif",
                'rest_pre_EC_StON': rest_folder / f"{subj_name}_rest_pre_EClosed_StON-epo.fif",
                'rest_pre_EC_StOFF': rest_folder / f"{subj_name}_rest_pre_EClosed_StOFF-epo.fif"
            }
        }

        # --- LOAD TASK DATA ---
        if condition == 'task':
            subj_data = {}
            for original_key, filepath in expected_files['task'].items():
                if not filepath.exists():
                    continue
                try:
                    epochs = mne.read_epochs(filepath, preload=True, verbose=False)
                    subj_data[original_key] = epochs
                except Exception as e:
                    print(f"  -> [ERROR] Failed to load {filepath.name}: {e}")
            
            if subj_data:
                master_group_data[subj_name] = subj_data

        # --- LOAD REST DATA ---
        elif condition == 'rest':
            subj_data = {'rest_pre': {}, 'rest_post': {}}
            for timepoint in ['rest_pre', 'rest_post']:
                for original_key, filepath in expected_files[timepoint].items():
                    if not filepath.exists():
                        continue
                    try:
                        target_key = key_translation[original_key]
                        epochs = mne.read_epochs(filepath, preload=True, verbose=False)
                        subj_data[timepoint][target_key] = epochs
                    except Exception as e:
                        print(f"  -> [ERROR] Failed to load {filepath.name}: {e}")
            
            # Only add the subject if at least one rest epoch file was successfully loaded
            if subj_data['rest_pre'] or subj_data['rest_post']:
                master_group_data[subj_name] = subj_data

    print(f"\nSuccessfully loaded '{condition}' data for {len(master_group_data)} subjects!")
    return master_group_data
