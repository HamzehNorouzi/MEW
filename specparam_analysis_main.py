import mne
from pathlib import Path
from utils.rest_spec import analyse_group_rest_specparam

master_group_data = {}

master_folder = Path('/home/hamzeh/Documents/MEAOW_project/pilot_data/preprocessed_files')

base_save_dir = master_folder
key_translation = {}
base_save_dir.mkdir(parents=True, exist_ok=True)

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


for subject_folder in master_folder.iterdir():
    if subject_folder.is_dir():
        subj_name = subject_folder.name
        print(f"\nLoading Rest Data for: {subj_name}...")

        subj_save_dir = base_save_dir / subj_name
        task_folder = subj_save_dir / 'task'
        rest_folder = subj_save_dir / 'rest'

        expected_files = {
            'task': {
                'gait': subj_save_dir / task_folder / f"{subj_name}_gait-epo.fif",
                'standup': subj_save_dir / task_folder / f"{subj_name}_standup-epo.fif"},

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

        master_group_data[subj_name] = subj_data

print(f"\nSuccessfully loaded data for {len(master_group_data)} subjects!")

group_results_df = analyse_group_rest_specparam(master_group_data, base_save_dir)
