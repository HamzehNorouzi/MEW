import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- Import from the older FOOOF package ---
from fooof import FOOOFGroup, FOOOF


def drop_bad_fit_epochs(epochs, condition_label, z_thresh=3.0):
    """
    Fits FOOOF to individual epochs and drops those with an outlier goodness-of-fit.
    """
    if len(epochs) < 3:
        print(f"  -> {condition_label}: Too few epochs ({len(epochs)}) for Z-score rejection. Skipping.")
        return epochs

    motor_rois = ['C3', 'C4', 'Cz', 'CP3', 'CP4', 'CPz', 'FC3', 'FC4', 'FCz']
    avail_chs = [ch for ch in motor_rois if ch in epochs.ch_names]

    # 1. Compute PSD per individual epoch
    spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=40, picks=avail_chs, verbose=False)
    freqs = spectrum.freqs

    # Average across channels (axis=1) to get one spectrum per epoch (n_epochs, n_freqs)
    psd_per_epoch = spectrum.get_data().mean(axis=1)

    # 2. Fit the FOOOFGroup Model to the 2D epoch data
    fg = FOOOFGroup(freq_range=[1, 40], aperiodic_mode='fixed', verbose=False)
    fg.fit(freqs, psd_per_epoch)

    # 3. Extract the Mean Absolute Error of the fit for every epoch
    fit_errors = fg.get_params('error')

    if np.std(fit_errors) == 0:
        return epochs

    # 4. Calculate Z-scores and drop outliers
    z_scores = zscore(fit_errors)
    good_idx = np.where(np.abs(z_scores) <= z_thresh)[0]

    n_dropped = len(epochs) - len(good_idx)
    print(
        f"  -> {condition_label}: FOOOF Fit Check dropped {n_dropped}/{len(epochs)} outlier epochs (|Z| > {z_thresh}).")

    return epochs[good_idx]


def analyze_rest_specparam(epochs_stim_off, epochs_stim_on, state_name):
    """
    Drops bad fit segments, compares StimOFF and StimON, computes PSDs, 
    fits classic FOOOF, and extracts aperiodic and periodic features.
    """
    print(f"\n--- Processing Resting State: {state_name} ---")

    # =========================================================
    # Drop outlier epochs based on Goodness-of-Fit Z-scores
    # =========================================================
    epochs_stim_off = drop_bad_fit_epochs(epochs_stim_off, "StimOFF")
    epochs_stim_on = drop_bad_fit_epochs(epochs_stim_on, "StimON")

    if len(epochs_stim_off) == 0 or len(epochs_stim_on) == 0:
        print("  -> Error: All epochs were rejected. Cannot complete analysis.")
        return None, None

    motor_rois = ['C3', 'C4', 'Cz', 'CP3', 'CP4', 'CPz', 'FC3', 'FC4', 'FCz']

    def get_roi_psd(epochs):
        avail_chs = [ch for ch in motor_rois if ch in epochs.ch_names]
        spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=40, picks=avail_chs)
        psd_data = spectrum.get_data().mean(axis=(0, 1))
        return spectrum.freqs, psd_data

    # 1. Get raw PSDs (1D arrays)
    freqs, psd_off = get_roi_psd(epochs_stim_off)
    _, psd_on = get_roi_psd(epochs_stim_on)

    # 2. Fit final 1D FOOOF Models
    print("Fitting final Grand Average FOOOF models...")
    fm_off = FOOOF(freq_range=[1, 40], aperiodic_mode='fixed')
    fm_on = FOOOF(freq_range=[1, 40], aperiodic_mode='fixed')

    fm_off.fit(freqs, psd_off)
    fm_on.fit(freqs, psd_on)

    # Extract Aperiodic Params using classic get_params()
    ap_off = fm_off.get_params('aperiodic_params')
    ap_on = fm_on.get_params('aperiodic_params')

    # Extract Flattened Spectra
    flat_spectra_off = fm_off.power_spectrum - fm_off._ap_fit
    flat_spectra_on = fm_on.power_spectrum - fm_on._ap_fit

    def extract_band_power(freq_array, spectra_array, fmin, fmax):
        idx = np.logical_and(freq_array >= fmin, freq_array <= fmax)
        return np.mean(spectra_array[idx])

        # 3. Generate Results Dictionary

    results = {
        'Condition': [f'{state_name}_StimOFF', f'{state_name}_StimON'],
        'Aperiodic_Offset': [ap_off[0], ap_on[0]],
        'Aperiodic_Exponent': [ap_off[1], ap_on[1]],
        'Raw_Power (1-12 Hz)': [extract_band_power(freqs, psd_off, 1, 12),
                                extract_band_power(freqs, psd_on, 1, 12)],
        'Raw_Power (12-35 Hz)': [extract_band_power(freqs, psd_off, 12, 35),
                                 extract_band_power(freqs, psd_on, 12, 35)],
        'Adj_Periodic_Power (1-12 Hz)': [extract_band_power(freqs, flat_spectra_off, 1, 12),
                                         extract_band_power(freqs, flat_spectra_on, 1, 12)],
        'Adj_Periodic_Power (12-35 Hz)': [extract_band_power(freqs, flat_spectra_off, 12, 35),
                                          extract_band_power(freqs, flat_spectra_on, 12, 35)],
    }

    df_results = pd.DataFrame(results)
    print("\n--- FOOOF & Power Results Table ---")
    print(df_results.to_markdown(index=False))

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(freqs, psd_off, label='Stim OFF', color='blue')
    axes[0].loglog(freqs, psd_on, label='Stim ON', color='red')
    axes[0].axvspan(1, 12, color='gray', alpha=0.2, label='1-12 Hz')
    axes[0].axvspan(12, 35, color='orange', alpha=0.2, label='12-35 Hz')
    axes[0].set_title(f'Raw Power Spectrum: {state_name}')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power (V^2/Hz)')
    axes[0].legend()

    axes[1].plot(freqs, flat_spectra_off, label='Stim OFF', color='blue')
    axes[1].plot(freqs, flat_spectra_on, label='Stim ON', color='red')
    axes[1].axvspan(1, 12, color='gray', alpha=0.2)
    axes[1].axvspan(12, 35, color='orange', alpha=0.2)
    axes[1].set_title(f'FOOOF Adjusted Periodic Power: {state_name}')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Log Power (Aperiodic Removed)')
    axes[1].legend()

    plt.tight_layout()
    return df_results, fig