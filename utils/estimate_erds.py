import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyse_group_task_erd(master_task_data, base_save_dir):
    """
    Computes single-epoch ERD/ERS for task data, separating by StimON and StimOFF.
    Extracts scalar values and time-series for Alpha and Beta bands.
    Plots TFR maps and specific condition comparisons.
    """
    print("\n=======================================================")
    print(" RUNNING GROUP-LEVEL STIM-SPECIFIC ERD/ERS ANALYSIS")
    print("=======================================================")

    # 1. Configuration
    rois = {
        'Left_Motor': ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5'],
        'Right_Motor': ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6']
    }

    # freqs = np.arange(1, 35, 0.5)
    freqs = np.logspace(*np.log10([8, 35]), num=50) # freqs = np.arange(4, 36, 1)
    n_cycles = freqs/2
    bands = {
        'Alpha': (7, 13),
        'Beta': (13, 35)
    }

    task_configs = {
        'standup': {'tmin': -3, 'tmax': 1, 'baseline': (-3, -2), 'task_window': (-1.5, 1)},
        'gait': {'tmin': -2.5, 'tmax': 2.0, 'baseline': (-2.0, -1), 'task_window': (-1, +1)}
    }

    # Map your specific MNE event_id strings to the clean labels
    task_event_map = {
        'standup': {'StimON': 'StimON_StandUp', 'StimOFF': 'StimOFF_StandUp'},
        'gait': {'StimON': 'StimON_Gait', 'StimOFF': 'StimOFF_Gait'}
    }

    # Data structures for group averaging
    group_tfrs = {task: {stim: {roi: [] for roi in rois} for stim in ['StimON', 'StimOFF']} for task in task_configs}
    group_timecourses = {
        task: {band: {stim: {roi: [] for roi in rois} for stim in ['StimON', 'StimOFF']} for band in bands} for task in
        task_configs}

    all_scalar_results = []
    all_timeseries_results = []
    task_times = {task: None for task in task_configs}

    # ========================================================
    # 2. PROCESS DATA DICTIONARY
    # ========================================================
    for subj_name, subj_data in master_task_data.items():
        print(f"\n--- Processing Subject: {subj_name} ---")

        for task_name in ['gait', 'standup']:
            epochs = subj_data.get(task_name)
            if epochs is None or len(epochs) == 0:
                continue

            config = task_configs[task_name]

            # Loop through StimON and StimOFF
            for stim_label, event_name in task_event_map[task_name].items():
                if event_name not in epochs.event_id:
                    print(f"  -> Missing {event_name} for {subj_name}. Skipping condition.")
                    continue

                # Isolate the specific condition
                epochs_cond = epochs[event_name].copy()
                if len(epochs_cond) == 0:
                    continue

                for roi_name, channels in rois.items():
                    avail_chs = [ch for ch in channels if ch in epochs_cond.ch_names]
                    if not avail_chs:
                        continue

                    epochs_roi = epochs_cond.copy().pick_channels(avail_chs)

                    # --- Compute TFR & ERD ---
                    epochs_roi.apply_baseline(config['baseline'], verbose=False)
                    tfr_epochs = mne.time_frequency.tfr_morlet(
                        epochs_roi, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        average=False, return_itc=False, n_jobs=-1, verbose=False
                    )
                    tfr_epochs.apply_baseline(config['baseline'], mode='percent', verbose=False)
                    subj_tfr_avg = tfr_epochs.average()

                    group_tfrs[task_name][stim_label][roi_name].append(subj_tfr_avg)

                    # --- Extract Band-Specific Signals ---
                    if task_times[task_name] is None:
                        task_times[task_name] = subj_tfr_avg.times

                    times_array = task_times[task_name]

                    for band_name, (fmin, fmax) in bands.items():
                        # Crop to frequency band and average across freqs & channels
                        tfr_band = subj_tfr_avg.copy().crop(fmin=fmin, fmax=fmax)
                        time_course = tfr_band.data.mean(axis=(0, 1)) * 100.0  # Convert to %

                        # Store for group plotting
                        group_timecourses[task_name][band_name][stim_label][roi_name].append(time_course)

                        # Extract single scalar for the active window
                        idx_min = np.searchsorted(times_array, config['task_window'][0])
                        idx_max = np.searchsorted(times_array, config['task_window'][1])
                        scalar_erd = time_course[idx_min:idx_max].mean()

                        all_scalar_results.append({
                            'Subject': subj_name, 'Task': task_name, 'Stim': stim_label,
                            'ROI': roi_name, 'Band': band_name, 'ERD_Percent': scalar_erd
                        })

                        # Save the full time-series for this subject/condition
                        for t, val in zip(times_array, time_course):
                            all_timeseries_results.append({
                                'Subject': subj_name, 'Task': task_name, 'Stim': stim_label,
                                'ROI': roi_name, 'Band': band_name, 'Time': t, 'ERD_Percent': val
                            })

                print(f"    -> {task_name} | {stim_label}: Processed {len(epochs_cond)} epochs.")

    # ========================================================
    # 3. PLOTTING GRAND AVERAGES
    # ========================================================
    print("\nComputing and Plotting Grand Averages...")
    group_save_dir = Path(base_save_dir) / 'Group_Results'
    group_save_dir.mkdir(exist_ok=True)

    def plot_with_sem(ax, data_list, times, color, label):
        """Helper to plot mean with shaded standard error."""
        if not data_list: return
        data_arr = np.vstack(data_list)
        mean = data_arr.mean(axis=0)
        sem = data_arr.std(axis=0) / np.sqrt(data_arr.shape[0])
        ax.plot(times, mean, color=color, label=label, linewidth=2)
        ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.2)

    for task_name in task_configs:
        baseline_times = task_configs[task_name]['baseline']
        times_array = task_times[task_name]

        # --- A. TFR Plots (2x2 Grid: Rows=ROI, Cols=Stim) ---
        fig_tfr, axes_tfr = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig_tfr.suptitle(f"{task_name.capitalize()} - Grand Average TFR", fontsize=16)

        plot_mapping = [
            (axes_tfr[0, 0], 'Left_Motor', 'StimON'), (axes_tfr[0, 1], 'Left_Motor', 'StimOFF'),
            (axes_tfr[1, 0], 'Right_Motor', 'StimON'), (axes_tfr[1, 1], 'Right_Motor', 'StimOFF')
        ]

        for ax, roi, stim in plot_mapping:
            if len(group_tfrs[task_name][stim][roi]) > 0:
                g_avg = mne.grand_average(group_tfrs[task_name][stim][roi])
                g_avg.plot(baseline=None, axes=ax, show=False, combine='mean', vlim=(-0.5, 0.5), cmap='RdBu_r')
                ax.set_title(f"{roi} | {stim}")
                ax.axvline(baseline_times[0], color='black', linestyle='--', alpha=0.7)
                ax.axvline(baseline_times[1], color='black', linestyle='--', alpha=0.7)

        plt.tight_layout()
        fig_tfr.savefig(group_save_dir / f"Group_TFR_{task_name}.png", dpi=300)
        plt.close(fig_tfr)

        # --- B. Band-Specific ERD Time Course Plots ---
        # Fulfills exact request: Left vs Right, ON vs OFF comparisons
        for band_name in bands.keys():
            fig_lines, axes_lines = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
            fig_lines.suptitle(f"{task_name.capitalize()} - {band_name} Band ERD/ERS (%)", fontsize=16)

            data = group_timecourses[task_name][band_name]

            # 1. Compare ON vs OFF in Left Hemisphere
            plot_with_sem(axes_lines[0, 0], data['StimON']['Left_Motor'], times_array, 'red', 'StimON')
            plot_with_sem(axes_lines[0, 0], data['StimOFF']['Left_Motor'], times_array, 'blue', 'StimOFF')
            axes_lines[0, 0].set_title("Left Hemisphere: ON vs OFF")

            # 2. Compare ON vs OFF in Right Hemisphere
            plot_with_sem(axes_lines[0, 1], data['StimON']['Right_Motor'], times_array, 'red', 'StimON')
            plot_with_sem(axes_lines[0, 1], data['StimOFF']['Right_Motor'], times_array, 'blue', 'StimOFF')
            axes_lines[0, 1].set_title("Right Hemisphere: ON vs OFF")

            # 3. Compare Left vs Right in StimOFF
            plot_with_sem(axes_lines[1, 0], data['StimOFF']['Left_Motor'], times_array, 'purple', 'Left Motor')
            plot_with_sem(axes_lines[1, 0], data['StimOFF']['Right_Motor'], times_array, 'green', 'Right Motor')
            axes_lines[1, 0].set_title("StimOFF: Left vs Right")

            # 4. Compare Left vs Right in StimON
            plot_with_sem(axes_lines[1, 1], data['StimON']['Left_Motor'], times_array, 'purple', 'Left Motor')
            plot_with_sem(axes_lines[1, 1], data['StimON']['Right_Motor'], times_array, 'green', 'Right Motor')
            axes_lines[1, 1].set_title("StimON: Left vs Right")

            for ax in axes_lines.flat:
                ax.axvline(0, color='black', linestyle='-')  # Movement onset
                ax.axvspan(baseline_times[0], baseline_times[1], color='gray', alpha=0.2, label='Baseline')
                ax.axhline(0, color='gray', linestyle='--')
                ax.set_ylabel("ERD/ERS (%)")
                ax.set_xlabel("Time (s)")
                ax.legend(loc='upper right')

            plt.tight_layout()
            fig_lines.savefig(group_save_dir / f"Group_TimeCourse_{task_name}_{band_name}.png", dpi=300)
            plt.close(fig_lines)

    # ========================================================
    # 4. SAVE DATAFRAMES
    # ========================================================
    df_scalars = pd.DataFrame(all_scalar_results)
    df_scalars.to_csv(group_save_dir / "Group_Task_ERD_Scalars.csv", index=False)

    df_timeseries = pd.DataFrame(all_timeseries_results)
    df_timeseries.to_csv(group_save_dir / "Group_Task_ERD_TimeSeries.csv", index=False)

    print(f"\nPipeline complete! Saved TFRs, Line Plots, and 2 CSVs to: {group_save_dir.name}/")
    return df_scalars, df_timeseries