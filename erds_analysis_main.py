import mne
from pathlib import Path
from utils.estimate_erds import analyse_group_task_erd
# Initialize the master dictionary for Task data
master_task_data = {}

master_folder = Path('/home/hamzeh/Documents/MEAOW_project/pilot_data/preprocessed_files')
base_save_dir = master_folder
base_save_dir.mkdir(parents=True, exist_ok=True)

for subject_folder in master_folder.iterdir():
    if subject_folder.is_dir():
        subj_name = subject_folder.name
        print(f"\nLoading Task Data for: {subj_name}...")

        # Re-establish paths for this specific subject
        subj_save_dir = base_save_dir / subj_name
        task_folder = subj_save_dir / 'task'

        # Your expected files dictionary for task
        expected_files = {
            'gait': task_folder / f"{subj_name}_gait-epo.fif",
            'standup': task_folder / f"{subj_name}_standup-epo.fif"
        }

        # Initialize the task dictionary for this specific subject
        subj_task_data = {}

        for task_key, filepath in expected_files.items():
            if not filepath.exists():
                print(f"  -> [WARNING] Missing file: {filepath.name}")
                continue

            try:
                # Load the epochs
                epochs = mne.read_epochs(filepath, preload=True, verbose=False)
                subj_task_data[task_key] = epochs
            except Exception as e:
                print(f"  -> [ERROR] Failed to load {filepath.name}: {e}")

        # Only add to the master dictionary if we actually loaded something
        if subj_task_data:
            master_task_data[subj_name] = subj_task_data

print(f"\nSuccessfully loaded task data for {len(master_task_data)} subjects!")

# Run the group analysis
group_erd_df = analyse_group_task_erd(master_task_data, base_save_dir)

