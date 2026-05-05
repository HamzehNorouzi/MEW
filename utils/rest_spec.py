import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from specparam import SpectralGroupModel, SpectralModel

'''
def analyse_group_rest_specparam(master_data_dict, save_dir, freq_range=[1, 100], z_thresh=3.0):
    """
    Computes Group-Level Grand Averages for resting state across Left and Right Motor ROIs.
    Drops outlier epochs based on Specparam Goodness-of-Fit (Error) Z-scores.
    Extracts raw and aperiodic-adjusted band powers.
    """
    print("\n=======================================================")
    print(" RUNNING GROUP-LEVEL SPECPARAM WITH OUTLIER REJECTION")
    print("=======================================================")

    # Define ROIs (All Left and Right C and CP channels)
    rois = {
        'Left_Motor': ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5'],
        'Right_Motor': ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6']
    }

    timepoints = ['rest_pre', 'rest_post']
    states = ['EOpen', 'EClosed']
    stim_conds = ['StON', 'StOFF']
    colors = {'StON': 'red', 'StOFF': 'blue'}

    # Specparam exactly as you defined them
    sp_kwargs = {
        'peak_width_limits': [0.5, 8], # zero must not be included in the peak_width_limits, y = h\e^{-{(x-c)^2}/{2w^2}}
        'max_n_peaks': 5,
        'min_peak_height': 0,
        'peak_threshold': 1,
        'aperiodic_mode': 'fixed',
        'verbose': False
    }

    def extract_band_power(freq_array, spectra_array, fmin, fmax):
        """Helper to extract mean power in a specific band"""
        idx = np.logical_and(freq_array >= fmin, freq_array <= fmax)
        return np.mean(spectra_array[idx])

    all_results = []

    # Process each ROI separately
    for roi_name, channels in rois.items():
        print(f"\n--- Processing ROI: {roi_name} ---")

        # Set up a figure for Raw Power (2x2) and Adjusted Power (2x2)
        fig_raw, axes_raw = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig_flat, axes_flat = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

        fig_raw.suptitle(f"Group Grand Average Raw Power - {roi_name}", fontsize=16)
        fig_flat.suptitle(f"Group FOOOF Adjusted Periodic Power - {roi_name}", fontsize=16)

        plot_map_raw = {('rest_pre', 'EOpen'): axes_raw[0, 0], ('rest_pre', 'EClosed'): axes_raw[0, 1],
                        ('rest_post', 'EOpen'): axes_raw[1, 0], ('rest_post', 'EClosed'): axes_raw[1, 1]}

        plot_map_flat = {('rest_pre', 'EOpen'): axes_flat[0, 0], ('rest_pre', 'EClosed'): axes_flat[0, 1],
                         ('rest_post', 'EOpen'): axes_flat[1, 0], ('rest_post', 'EClosed'): axes_flat[1, 1]}

        # Formatting subplots
        for axes, fig_type in zip([axes_raw, axes_flat], ['Raw Power', 'Adj Power']):
            axes[0, 0].set_title("PRE - Eye Open")
            axes[0, 1].set_title("PRE - Eye Closed")
            axes[1, 0].set_title("POST - Eye Open")
            axes[1, 1].set_title("POST - Eye Closed")
            for ax in axes.flat:
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (V^2/Hz)' if fig_type == 'Raw Power' else 'Log Power (Aperiodic Removed)')
                ax.axvspan(1, 5, color='gray', alpha=0.1, label='1-5 Hz')
                ax.axvspan(7, 13, color='green', alpha=0.1, label='7-13 Hz')
                ax.axvspan(14, 35, color='orange', alpha=0.1, label='14-35 Hz')

        for timepoint in timepoints:
            for state in states:
                ax_raw = plot_map_raw[(timepoint, state)]
                ax_flat = plot_map_flat[(timepoint, state)]

                for stim in stim_conds:
                    condition_key = f"{state}_{stim}"
                    all_psds = []
                    freqs = None

                    # 1. Pool all epochs across all subjects
                    for subj, subj_data in master_data_dict.items():
                        if timepoint in subj_data and condition_key in subj_data[timepoint]:
                            epochs = subj_data[timepoint][condition_key]

                            if epochs is not None and len(epochs) > 0:
                                avail_chs = [ch for ch in channels if ch in epochs.ch_names]
                                if not avail_chs:
                                    continue

                                spectrum = epochs.compute_psd(method='welch', fmin=freq_range[0], fmax=freq_range[1],
                                                              picks=avail_chs, n_jobs=-1, verbose=False)
                                if freqs is None:
                                    freqs = spectrum.freqs

                                # Average over channels, keeping epochs separate for outlier detection
                                psds_per_epoch = spectrum.get_data().mean(axis=1)
                                all_psds.append(psds_per_epoch)

                    if not all_psds:
                        continue

                    group_psds_2d = np.vstack(all_psds)
                    n_total_epochs = group_psds_2d.shape[0]

                    # 2. Outlier Rejection via SpectralGroupModel
                    fg = SpectralGroupModel(**sp_kwargs)
                    fg.fit(freqs, group_psds_2d, freq_range=freq_range)

                    fit_errors = np.array([res.metrics['gof_rsquared'] for res in fg.results.group_results])
                    error_zscores = zscore(fit_errors)

                    # Keep epochs where Z-score is within threshold
                    clean_mask = np.abs(error_zscores) < z_thresh
                    clean_group_psds = group_psds_2d[clean_mask]
                    dropped_count = n_total_epochs - len(clean_group_psds)
                    print(
                        f"[{roi_name} | {timepoint} | {condition_key}] Dropped {dropped_count} outlier epochs (Z > {z_thresh}). Kept {len(clean_group_psds)}.")

                    # 3. Grand Average & Final FOOOF Model
                    grand_avg_psd = clean_group_psds.mean(axis=0)

                    fm = SpectralModel(**sp_kwargs)
                    fm.fit(freqs, grand_avg_psd, freq_range=freq_range)

                    # Extract Aperiodic components
                    ap_params = fm.get_params('aperiodic')
                    apfit = ap_params[0] - np.log10(freqs ** ap_params[1])
                    flat_spectrum = fm.data.power_spectrum - apfit

                    # Extract Frequency Bands
                    raw_bands = {
                        'Raw_1_5_Hz': extract_band_power(freqs, grand_avg_psd, 1, 5),
                        'Raw_7_13_Hz': extract_band_power(freqs, grand_avg_psd, 7, 13),
                        'Raw_14_35_Hz': extract_band_power(freqs, grand_avg_psd, 14, 35)
                    }

                    adj_bands = {
                        'Adj_1_5_Hz': extract_band_power(freqs, flat_spectrum, 1, 5),
                        'Adj_7_13_Hz': extract_band_power(freqs, flat_spectrum, 7, 13),
                        'Adj_14_35_Hz': extract_band_power(freqs, flat_spectrum, 14, 35)
                    }

                    # Store Data
                    all_results.append({
                        'ROI': roi_name,
                        'Timepoint': timepoint,
                        'Eye_State': state,
                        'Stim_Condition': stim,
                        'Total_Epochs': n_total_epochs,
                        'Outliers_Dropped': dropped_count,
                        'Aperiodic_Offset': ap_params[0],
                        'Aperiodic_Exponent': ap_params[1],
                        'Group_R_squared': fm.results.metrics['gof_rsquared'],
                        **raw_bands,
                        **adj_bands
                    })

                    # Plotting Overlays (Red/Blue)
                    color = colors[stim]

                    # Raw PSD Plot
                    ax_raw.loglog(freqs, grand_avg_psd, label=f'Stim {stim.replace("St", "")}', color=color)

                    # Flattened PSD Plot
                    ax_flat.plot(freqs, flat_spectrum, label=f'Stim {stim.replace("St", "")}', color=color)

                ax_raw.legend(loc='lower left', fontsize='small')
                ax_flat.legend(loc='upper right', fontsize='small')

        plt.tight_layout()

        # Save plots
        fig_raw.savefig(save_dir / f"Group_GrandAvg_Raw_{roi_name}.eps", dpi=1200)
        fig_flat.savefig(save_dir / f"Group_GrandAvg_Flat_{roi_name}.eps", dpi=1200)
        plt.close(fig_raw)
        plt.close(fig_flat)

    # Save DataFrame
    df_results = pd.DataFrame(all_results)
    csv_path = save_dir / "Group_Specparam_Band_Powers.csv"
    df_results.to_csv(csv_path, index=False)

    print("\n--- FOOOF & Power Results Table (Sample) ---")
    print(df_results.head().to_markdown(index=False))
    print(f"\nSaved master results to {csv_path.name}")

    return df_results
'''


# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from specparam import SpectralModel, SpectralGroupModel  # Assuming fooof/specparam is installed




def analyse_group_rest_specparam(master_data_dict, save_dir, freq_range=[1, 100], z_thresh=3.0):
    """
    Computes Subject-Level Specparam/FOOOF features for resting state across ROIs.
    Pools epochs for group-level outlier rejection based on GoF Z-scores,
    then extracts raw and aperiodic-adjusted band powers PER SUBJECT.
    Returns a dataframe ready for repeated-measures statistics.
    """
    print("\n=======================================================")
    print(" RUNNING SUBJECT-LEVEL SPECPARAM WITH OUTLIER REJECTION")
    print("=======================================================")

    # Ensure save directory is a Path object and exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define ROIs (All Left and Right C and CP channels)
    rois = {
        'Left_Motor': ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5'],
        'Right_Motor': ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6']
    }

    timepoints = ['rest_pre', 'rest_post']
    states = ['EOpen', 'EClosed']
    stim_conds = ['StON', 'StOFF']
    colors = {'StON': 'red', 'StOFF': 'blue'}

    # Specparam parameters
    sp_kwargs = {
        'peak_width_limits': [0.5, 8],
        'max_n_peaks': 5,
        'min_peak_height': 0,
        'peak_threshold': 1,
        'aperiodic_mode': 'fixed',
        'verbose': False
    }

    def extract_band_power(freq_array, spectra_array, fmin, fmax):
        """Helper to extract mean power in a specific band"""
        idx = np.logical_and(freq_array >= fmin, freq_array <= fmax)
        return np.mean(spectra_array[idx])

    all_results = []

    # Process each ROI separately
    for roi_name, channels in rois.items():
        print(f"\n--- Processing ROI: {roi_name} ---")

        # Set up a figure for Raw Power (2x2) and Adjusted Power (2x2)
        fig_raw, axes_raw = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig_flat, axes_flat = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

        fig_raw.suptitle(f"Group Grand Average Raw Power - {roi_name}", fontsize=16)
        fig_flat.suptitle(f"Group FOOOF Adjusted Periodic Power - {roi_name}", fontsize=16)

        plot_map_raw = {('rest_pre', 'EOpen'): axes_raw[0, 0], ('rest_pre', 'EClosed'): axes_raw[0, 1],
                        ('rest_post', 'EOpen'): axes_raw[1, 0], ('rest_post', 'EClosed'): axes_raw[1, 1]}

        plot_map_flat = {('rest_pre', 'EOpen'): axes_flat[0, 0], ('rest_pre', 'EClosed'): axes_flat[0, 1],
                         ('rest_post', 'EOpen'): axes_flat[1, 0], ('rest_post', 'EClosed'): axes_flat[1, 1]}

        # Formatting subplots
        for axes, fig_type in zip([axes_raw, axes_flat], ['Raw Power', 'Adj Power']):
            axes[0, 0].set_title("PRE - Eye Open")
            axes[0, 1].set_title("PRE - Eye Closed")
            axes[1, 0].set_title("POST - Eye Open")
            axes[1, 1].set_title("POST - Eye Closed")
            for ax in axes.flat:
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (V^2/Hz)' if fig_type == 'Raw Power' else 'Log Power (Aperiodic Removed)')
                ax.axvspan(1, 5, color='gray', alpha=0.1, label='1-5 Hz')
                ax.axvspan(7, 13, color='green', alpha=0.1, label='7-13 Hz')
                ax.axvspan(14, 35, color='orange', alpha=0.1, label='14-35 Hz')

        for timepoint in timepoints:
            for state in states:
                ax_raw = plot_map_raw[(timepoint, state)]
                ax_flat = plot_map_flat[(timepoint, state)]

                for stim in stim_conds:
                    condition_key = f"{state}_{stim}"
                    all_psds = []
                    epoch_subj_labels = []  # NEW: Keep track of which subject owns which epoch
                    freqs = None

                    # 1. Pool all epochs across all subjects, but track subject IDs
                    for subj, subj_data in master_data_dict.items():
                        if timepoint in subj_data and condition_key in subj_data[timepoint]:
                            epochs = subj_data[timepoint][condition_key]

                            if epochs is not None and len(epochs) > 0:
                                avail_chs = [ch for ch in channels if ch in epochs.ch_names]
                                if not avail_chs:
                                    continue

                                spectrum = epochs.compute_psd(method='welch', fmin=freq_range[0], fmax=freq_range[1],
                                                              picks=avail_chs, n_jobs=-1, verbose=False)
                                if freqs is None:
                                    freqs = spectrum.freqs

                                # Average over channels, keeping epochs separate
                                psds_per_epoch = spectrum.get_data().mean(axis=1)
                                all_psds.append(psds_per_epoch)

                                # Tag each epoch with its subject ID
                                epoch_subj_labels.extend([subj] * len(psds_per_epoch))

                    if not all_psds:
                        continue

                    group_psds_2d = np.vstack(all_psds)
                    epoch_subj_labels = np.array(epoch_subj_labels)
                    n_total_epochs = group_psds_2d.shape[0]

                    # 2. Outlier Rejection via SpectralGroupModel on POOLED data
                    fg = SpectralGroupModel(**sp_kwargs)
                    fg.fit(freqs, group_psds_2d, freq_range=freq_range)

                    fit_errors = np.array([res.metrics['gof_rsquared'] for res in fg.results.group_results])
                    error_zscores = zscore(fit_errors)

                    # Create a mask of "clean" epochs
                    clean_mask = np.abs(error_zscores) < z_thresh
                    clean_group_psds = group_psds_2d[clean_mask]
                    clean_subj_labels = epoch_subj_labels[clean_mask]

                    dropped_count = n_total_epochs - len(clean_group_psds)
                    print(
                        f"[{roi_name} | {timepoint} | {condition_key}] Dropped {dropped_count} outlier epochs globally (Z > {z_thresh}).")

                    # 3. NEW: Extract Subject-Level Features
                    unique_subjects = np.unique(clean_subj_labels)

                    # Accumulators to calculate Grand Average for plotting later
                    grand_avg_raw_accumulator = []
                    grand_avg_flat_accumulator = []

                    for subj in unique_subjects:
                        # Isolate the clean PSDs for THIS subject only
                        subj_mask = (clean_subj_labels == subj)
                        subj_clean_psds = clean_group_psds[subj_mask]

                        # Calculate the subject's average PSD
                        subj_avg_psd = subj_clean_psds.mean(axis=0)

                        # Fit FOOOF Model specifically for this subject
                        fm = SpectralModel(**sp_kwargs)
                        fm.fit(freqs, subj_avg_psd, freq_range=freq_range)

                        # Extract Aperiodic components and flatten spectrum
                        ap_params = fm.get_params('aperiodic')
                        apfit = ap_params[0] - np.log10(freqs ** ap_params[1])
                        flat_spectrum = fm.data.power_spectrum - apfit

                        # Extract Frequency Bands
                        raw_bands = {
                            'Raw_1_5_Hz': extract_band_power(freqs, subj_avg_psd, 1, 5),
                            'Raw_7_13_Hz': extract_band_power(freqs, subj_avg_psd, 7, 13),
                            'Raw_14_35_Hz': extract_band_power(freqs, subj_avg_psd, 14, 35)
                        }

                        adj_bands = {
                            'Adj_1_5_Hz': extract_band_power(freqs, flat_spectrum, 1, 5),
                            'Adj_7_13_Hz': extract_band_power(freqs, flat_spectrum, 7, 13),
                            'Adj_14_35_Hz': extract_band_power(freqs, flat_spectrum, 14, 35)
                        }

                        # Store Subject-Level Data
                        all_results.append({
                            'Subject': subj,
                            'ROI': roi_name,
                            'Timepoint': timepoint,
                            'Eye_State': state,
                            'Stim_Condition': stim,
                            'Subject_Clean_Epochs': len(subj_clean_psds),
                            'Aperiodic_Offset': ap_params[0],
                            'Aperiodic_Exponent': ap_params[1],
                            'Subject_R_squared': fm.results.metrics['gof_rsquared'],
                            **raw_bands,
                            **adj_bands
                        })

                        # Add to accumulator for plotting
                        grand_avg_raw_accumulator.append(subj_avg_psd)
                        grand_avg_flat_accumulator.append(flat_spectrum)

                    # 4. Plotting Overlays (Red/Blue)
                    color = colors[stim]

                    # Calculate the true Grand Average (average of subject averages)
                    grand_avg_psd_final = np.mean(grand_avg_raw_accumulator, axis=0)
                    grand_avg_flat_final = np.mean(grand_avg_flat_accumulator, axis=0)

                    # Calculate the Standard Error of the Mean (SEM)
                    sem_raw = sem(grand_avg_raw_accumulator, axis=0)
                    sem_flat = sem(grand_avg_flat_accumulator, axis=0)

                    # Raw PSD Plot with SEM shading
                    ax_raw.loglog(freqs, grand_avg_psd_final, label=f'Stim {stim.replace("St", "")}', color=color)
                    # For loglog scales, prevent the lower bound of the SEM from hitting <= 0
                    lower_bound_raw = np.maximum(grand_avg_psd_final - sem_raw, 1e-15)
                    ax_raw.fill_between(freqs, lower_bound_raw, grand_avg_psd_final + sem_raw, color=color, alpha=0.2)

                    # Flattened PSD Plot with SEM shading
                    ax_flat.plot(freqs, grand_avg_flat_final, label=f'Stim {stim.replace("St", "")}', color=color)
                    ax_flat.fill_between(freqs, grand_avg_flat_final - sem_flat, grand_avg_flat_final + sem_flat,
                                         color=color, alpha=0.2)

                ax_raw.legend(loc='lower left', fontsize='small')
                ax_flat.legend(loc='upper right', fontsize='small')

        plt.tight_layout()

        # Save plots
        fig_raw.savefig(save_dir / f"Group_GrandAvg_Raw_{roi_name}.eps", dpi=1200)
        fig_flat.savefig(save_dir / f"Group_GrandAvg_Flat_{roi_name}.eps", dpi=1200)
        plt.close(fig_raw)
        plt.close(fig_flat)

    # Save DataFrame
    df_results = pd.DataFrame(all_results)

    # Save the updated dataframe to disk!
    csv_path = save_dir / "SubjectLevel_Specparam_Band_Powers.csv"
    df_results.to_csv(csv_path, index=False)

    print("\n--- FOOOF & Power Results Table (Sample) ---")
    print(df_results.head().to_markdown(index=False))
    print(f"\nSaved master subject-level results to {csv_path}")

    return df_results