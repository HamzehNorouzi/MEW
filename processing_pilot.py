import os.path
from unittest import main

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils.load_multiple_raw import  load_and_concat
from utils.preprocess import preprocess_eeg
from utils.estimate_erds import analyse_group_task_erd
from utils.rest_spec import analyse_group_rest_specparam
import json

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    print('Processing data initiated. \nPlease be patient!...')

    master_folder = Path('/home/hamzeh/Documents/MEAOW_project/pilot_data')
    montage_path = '//home/hamzeh/Documents/MEAOW_project/pilot_data/standard_waveguard64.elc'
    file_extension = '.vhdr'
    base_save_dir = master_folder / 'preprocessed_files'
    base_save_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = base_save_dir / 'preprocessing_logs_summary.json'

    preprocessing_stats = {
        'number_of_bad_components': {},
        'bad_subjects': [],
        'rejected_epochs': {}
    }

    for subject_folder in master_folder.iterdir():
        if subject_folder.is_dir():
            subj_name = subject_folder.name
            print(f"\n{'=' * 50}\nProcessing Subject: {subj_name}\n{'=' * 50}")

            subj_save_dir = base_save_dir / subj_name
            task_folder = subj_save_dir / 'task'
            rest_folder = subj_save_dir / 'rest'

            subj_save_dir.mkdir(exist_ok=True)
            task_folder.mkdir(exist_ok=True)
            rest_folder.mkdir(exist_ok=True)

            clean_epochs_dict = {'task': {'gait': None, 'standup': None}, 'rest_pre': None, 'rest_post': None}

            expected_files = {

                'task' : {
                'gait': subj_save_dir / task_folder / f"{subj_name}_gait-epo.fif",
                'standup': subj_save_dir/ task_folder / f"{subj_name}_standup-epo.fif"},

                'rest_post' : {
                'rest_post_EO_StON': subj_save_dir/ rest_folder / f"{subj_name}_rest_post_EOpen_StON-epo.fif",
                'rest_post_EO_StOFF': subj_save_dir/ rest_folder / f"{subj_name}_rest_post_EOpen_StOFF-epo.fif",
                'rest_post_EC_StON': subj_save_dir/ rest_folder / f"{subj_name}_rest_post_EClosed_StON-epo.fif",
                'rest_post_EC_StOFF': subj_save_dir/ rest_folder / f"{subj_name}_rest_post_EClosed_StOFF-epo.fif"},

                'rest_pre': {
                'rest_pre_EO_StON': subj_save_dir/ rest_folder / f"{subj_name}_rest_pre_EOpen_StON-epo.fif",
                'rest_pre_EO_StOFF': subj_save_dir/ rest_folder / f"{subj_name}_rest_pre_EOpen_StOFF-epo.fif",
                'rest_pre_EC_StON': subj_save_dir/ rest_folder / f"{subj_name}_rest_pre_EClosed_StON-epo.fif",
                'rest_pre_EC_StOFF': subj_save_dir/ rest_folder / f"{subj_name}_rest_pre_EClosed_StOFF-epo.fif"}}

            conditions_to_run = ['task', 'rest_pre', 'rest_post']

            needs_preprocessing = []

            for condition in conditions_to_run:
                print(f"\nChecking {condition} data...")
                files_for_cond = expected_files[condition]

                all_exist = all(filepath.exists() for filepath in files_for_cond.values())

                if all_exist:
                    print(f"  -> {len(files_for_cond)} processed files found for {condition}. Loading...")
                    clean_epochs_dict[condition] = {}
                    for sub_cond, filepath in files_for_cond.items():
                        clean_epochs_dict[condition][sub_cond] = mne.read_epochs(filepath, preload=True, verbose=False)

                else:
                    print(f"  -> MISSING files for {condition}. Will run preprocessing pipeline.")
                    needs_preprocessing.append(condition)
                    raw_folder = subject_folder / ('task' if condition == 'task' else 'rest')
                    raw_files = []
                    if raw_folder.exists():
                        for f in raw_folder.iterdir():
                            if f.suffix == file_extension:
                                fname_lower = f.name.lower()
                                if condition == 'task' and 'task' in fname_lower:
                                    raw_files.append(f)
                                elif condition == 'rest_pre' and 'pre' in fname_lower:
                                    raw_files.append(f)
                                elif condition == 'rest_post' and 'post' in fname_lower:
                                    raw_files.append(f)

                    raw_files = sorted(raw_files)

                    if not raw_files:
                        print(f"  -> [ERROR] No raw files found for {condition} in {raw_folder}!")
                        continue

                    raw_data = load_and_concat(raw_files)

                    if raw_data is not None:
                        clean_data = preprocess_eeg(raw_data, subj_name, condition, montage_path, preprocessing_stats)

                        if clean_data is not None:
                            if condition == 'task':
                                for task_name, epochs_obj in clean_data.items():
                                    if epochs_obj is not None:
                                        save_file = subj_save_dir / f"{subj_name}_{task_name}-epo.fif"
                                        print(f"Saving {task_name} epochs to {save_file}...")
                                        epochs_obj.save(save_file, overwrite=True)
                                        print(f"  --> IMMEDIATE SAVE SUCCESSFUL: {save_file.name}")
                                clean_epochs_dict[condition] = clean_data
                            else:
                                clean_epochs_dict[condition] = clean_data
                                for rest_cond, rest_epochs in clean_data.items():
                                    save_path = subj_save_dir / f"{subj_name}_{condition}_{rest_cond}-epo.fif"
                                    rest_epochs.save(save_path, overwrite=True)
                                    print(f"  --> IMMEDIATE SAVE SUCCESSFUL: {save_path.name}")
                        else:
                            print(f"  -> [WARNING] Preprocessing returned nothing for {subj_name} in {condition} .")

                    print("Saving preprocessing logs...")
                    with open(log_file_path, 'w') as json_file:
                        json.dump(preprocessing_stats, json_file, indent=4, cls=NumpyEncoder)
                    print(f"Logs successfully saved to: {log_file_path}")
            print(' All Files Loaded! ')
