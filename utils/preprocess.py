import mne
import numpy as np
import scipy.signal
import asrpy
from mne.preprocessing import ICA
from mne.viz import plot_montage
from mne_icalabel import label_components
from autoreject import AutoReject
from mne.preprocessing import create_eog_epochs
from sympy.plotting.pygletplot import plot
from mne.transforms import rotation
from utils.extract_rest_states import extract_continuous_blocks
def preprocess_eeg(raw, subject_name, condition_type, montage_path, stats_tracker):
    """
    Preprocesses raw EEG data, runs ASR, ICA (ICLabel), epochs the data, 
    and cleans epochs using Autoreject.
    
    Parameters:
    - raw: mne.io.Raw object
    - subject_name: str, name of the subject
    - condition_type: str, either 'task' or 'rest'
    - montage_path: str, path to the .elec montage file
    - stats_tracker: dict, used to track rejected ICs, bad subjects, and dropped epochs
    """
    print(f"\n--- Preprocessing {subject_name} | Condition: {condition_type} ---")

    # 0. Drop Mastoids and downsample 250 Hz
    raw.load_data()  # Data must be loaded into memory to apply functions
    print("Dropping mastoid channels...")
    channels_to_drop = [ch for ch in ['M1', 'M2'] if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    raw.resample(250)
    # ---------------------------------------------------------
    # 1. Detrend the Data
    # ---------------------------------------------------------
    print("Detrending data...")
    raw.apply_function(scipy.signal.detrend, channel_wise=True)

    # ---------------------------------------------------------
    # 2. High-Pass Filter (>1Hz)
    # ---------------------------------------------------------
    print("Applying high-pass'EOpen_StimOFF' in clean_pre filter (1 Hz)...")
    raw.filter(
        l_freq=1.0,
        h_freq=100.0,
        method='fir',
        fir_design='firwin',
        phase='zero'  # Forward-backward pass to eliminate phase shift
    )

    # ---------------------------------------------------------
    # 3. Common Average Reference (CAR)
    # ---------------------------------------------------------
    print("Applying Common Average Reference...")
    raw.set_eeg_reference('average', projection=False)

    # ---------------------------------------------------------
    # 4. Load and Apply Montage
    # ---------------------------------------------------------
    print("Applying montage...")
    if 'EOG' in raw.ch_names:
        raw.set_channel_types({'EOG': 'eog'})
    montage = mne.channels.read_custom_montage(montage_path)
    ch_pos = montage.get_positions()['ch_pos']
    # Create a new dictionary for rotated positions/ montage requires +90 rotation
    rotated_ch_pos = {}
    for ch_name, pos in ch_pos.items():
        # pos is [x, y, z]
        # To rotate +90 deg: new_x = -y, new_y = x
        new_pos = np.array([-pos[1], pos[0], pos[2]])
        rotated_ch_pos[ch_name] = new_pos

    rotated_montage = mne.channels.make_dig_montage(
        ch_pos=rotated_ch_pos,
        coord_frame=montage.get_positions()['coord_frame']
    )
    raw.set_montage(rotated_montage, match_case=False, on_missing='warn', verbose=False)

    # ---------------------------------------------------------
    # 5. ASRpy (Artifact Subspace Reconstruction)
    # ---------------------------------------------------------
    print("Running ASR...")
    asr = asrpy.ASR(sfreq=raw.info['sfreq'])
    asr.fit(raw)
    raw = asr.transform(raw)

    # ---------------------------------------------------------
    # 6 & 7. ICA & ICLabel
    # ---------------------------------------------------------
    print("Running ICA (Infomax)...")
    # Using 'infomax' instead of fastica. Picard is also a great fast alternative.
    ica = ICA(n_components=60, method='infomax', fit_params=dict(extended=True), random_state=42)
    ica.fit(raw)

    # Eye movement artifact
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG', verbose=False)

    # plotting eye movements artifact
    # eog_evoked = create_eog_epochs(raw, ch_name='EOG').average()
    # ica.plot_overlay(eog_evoked, exclude=eog_indices)

    print("Running ICLabel...")
    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels["labels"]

    # Identify non-brain components
    bad_ics = [i for i, label in enumerate(labels) if label != "brain"]
    bad_ics.extend(eog_indices)
    bad_ics = np.unique(bad_ics)
    num_bad_ics = len(bad_ics)

    # Save IC stats for this subject
    stats_tracker['number_of_bad_components'][subject_name] = num_bad_ics
    print(f"Rejected {num_bad_ics}/63 ICs.")

    # Exclude subject if >50% ICs are bad
    if num_bad_ics > (60 * 0.9):
        print(f"!!! WARNING: >50% ICs rejected. Excluding {subject_name}. !!!")
        if subject_name not in stats_tracker['bad_subjects']:
            stats_tracker['bad_subjects'].append(subject_name)
        return None # Stop processing and return nothing for this file

    # Apply ICA exclusion
    ica.exclude = bad_ics
    raw = ica.apply(raw)

    # ---------------------------------------------------------
    # 8. Define Events and Epoching
    # ---------------------------------------------------------
    print("Renaming annotations for epoching...")
    
    # BrainVision markers usually include prefixes like 'Stimulus/S2212'.
    # We will search the annotations for the target numbers and rename them.
    if condition_type == 'task':
        # 1. Define configurations for each task in one place
        task_configs = {
            'StandUp': {
                'event_map': {
                    'StimON_StandUp': ['Stimulus/s2212', 'Stimulus/s2222'],
                    'StimOFF_StandUp': ['Stimulus/s1212', 'Stimulus/s1222']
                },
                'tmin': -1.5, 'tmax': 0.3, 'baseline': (0, 0.3)
            },
            'Gait': {
                'event_map': {
                    'StimON_Gait': ['Stimulus/s2213', 'Stimulus/s2223'],
                    'StimOFF_Gait': ['Stimulus/s1223', 'Stimulus/s1213']
                },
                'tmin': -0.2, 'tmax': 1.0, 'baseline': (-0.2, 0)
            }
        }

        # 2. Combine all event maps into one for a SINGLE pass over the annotations
        combined_map = {**task_configs['StandUp']['event_map'], **task_configs['Gait']['event_map']}
        print("Renaming task annotations...")
        for i, annot in enumerate(raw.annotations):
            for new_label, triggers in combined_map.items():
                if any(trig in annot['description'] for trig in triggers):
                    raw.annotations.description[i] = new_label

        # 3. Extract events from MNE just ONCE
        events, event_id = mne.events_from_annotations(raw, event_id=None)

        # Dictionary to hold our final extracted epochs
        extracted_epochs = {}

        # 4. Loop through our configurations to epoch StandUp and Gait automatically
        for task_name, config in task_configs.items():
            # Filter for only the events needed for THIS specific task
            target_ids = {k: v for k, v in event_id.items() if k in config['event_map']}

            if not target_ids:
                print(f"Warning: No valid {task_name} events found for {subject_name}. Skipping {task_name}.")
                extracted_epochs[task_name] = None
                continue

            print(f"Epoching {task_name} data...")
            epochs = mne.Epochs(
                raw, events, event_id=target_ids,
                tmin=config['tmin'], tmax=config['tmax'],
                baseline=config['baseline'], preload=True
            )

            # Apply detrending
            epochs.apply_function(scipy.signal.detrend, channel_wise=True)

            # Apply Autoreject
            print("Running Autoreject...")
            n_epochs = len(epochs)
            # Autoreject needs at least 10 epochs to do cross-validation.
            if n_epochs < 10:
                print(f"Warning: Only {n_epochs} epoch(s) found. Skipping Autoreject.")
                dropped_counts = {cond: 0 for cond in target_ids.keys()}
                file_key = f"{subject_name}_{condition_type}"
                stats_tracker['rejected_epochs'][file_key] = dropped_counts
                return epochs  # Return the uncleaned epochs since we can't run AR

            # Dynamically set the number of cross-validation folds
            # Use 10 if we have plenty of epochs, otherwise use the number of epochs we have
            cv_splits = min(10, n_epochs)

            ar = AutoReject(n_jobs=-1, cv=cv_splits)
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

            # Calculate dropped epochs per condition
            rejected_bools = reject_log.bad_epochs
            dropped_counts = {cond: 0 for cond in target_ids.keys()}

            for idx, is_bad in enumerate(rejected_bools):
                if is_bad:
                    event_code = epochs.events[idx, 2]
                    for cond_name, cond_code in target_ids.items():
                        if cond_code == event_code:
                            dropped_counts[cond_name] += 1

            file_key = f"{subject_name}_{condition_type}"
            stats_tracker['rejected_epochs'][file_key] = dropped_counts
            print(f"Autoreject complete. Dropped epochs per condition: {dropped_counts}")

            extracted_epochs[task_name] = epochs_clean

        # Unpack the dictionary into your specific variables if you need them downstream
        StandUp_epochs = extracted_epochs.get('StandUp')
        Gait_epochs = extracted_epochs.get('Gait')

        task_epochs = {'gait': Gait_epochs, 'standup': StandUp_epochs}
        return task_epochs

    else: # Rest (pre or post)
        event_map = {
            'EOpen_StON': ['Stimulus/s1003'],
            'EClosed_StON': ['Stimulus/s1004'],
            'EOpen_StOFF': ['Stimulus/s1001'],
            'EClosed_StOFF': ['Stimulus/s1002']
        }

        print("Extracting continuous resting blocks (Minimum duration: 60s)...")
        # 1. Call the helper function!
        continuous_blocks = extract_continuous_blocks(raw, event_map, min_duration=90)
        if not continuous_blocks:
            print(f"Warning: No valid rest blocks found for {subject_name}.")
            return None

        final_rest_epochs = {}

        # 2. Chop into 2s windows and run Autoreject
        for label, combined_raw in continuous_blocks.items():
            print(f"Creating 2-second fixed epochs for {label}...")
            epochs_2s = mne.make_fixed_length_epochs(combined_raw, duration=2.0, preload=True)

            n_epochs_2s = len(epochs_2s)
            if n_epochs_2s >= 2:
                print(f"Running Autoreject on {label}...")
                cv_splits = min(10, n_epochs_2s)
                ar = AutoReject(n_jobs=-1, cv=cv_splits)
                epochs_clean, reject_log = ar.fit_transform(epochs_2s, return_log=True)

                dropped_count = sum(reject_log.bad_epochs)
                stats_tracker['rejected_epochs'][f"{subject_name}_{condition_type}_{label}"] = dropped_count

                final_rest_epochs[label] = epochs_clean
            else:
                print(f"Warning: {label} has fewer than 2 epochs. Skipping Autoreject.")
                final_rest_epochs[label] = epochs_2s
                stats_tracker['rejected_epochs'][f"{subject_name}_{condition_type}_{label}"] = 0

        print(f"Successfully created 2s epochs for: {list(final_rest_epochs.keys())}")
        return final_rest_epochs
