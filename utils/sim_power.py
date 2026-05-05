import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from specparam.sim import sim_power_spectrum, sim_group_power_spectra, gen_freqs
from specparam.plts.spectra import plot_spectra
from specparam import SpectralModel

def test_pipeline_with_specparam_sim():
    print("Generating mathematically perfect spectra using specparam.sim...")

    # ====================================================================
    # 1. SIMULATE SPECTRA
    # aperiodic_params = [offset, exponent]
    # periodic_params = [[center_freq_1, height_1, bandwidth_1], [freq_2...]]
    # ====================================================================
    fre_range = [1, 40]
    # StimOFF: Strong Alpha (10Hz, height=0.8), Weak Beta (22Hz, height=0.2)
    aperiodic_params = [1, 500, 2]
    periodic_params = [9, 0.4, 1, 24, 0.2, 3]

    freqs, psd_off = sim_power_spectrum(fre_range,
     {'knee' : aperiodic_params},
     {'gaussian' : periodic_params},
      nlv=0.02, freq_res=0.5)

    # StimON: Weak Alpha (10Hz, height=0.2), Strong Beta (22Hz, height=0.6)
    freqs, psd_on= sim_power_spectrum(fre_range,
                                 {'knee': aperiodic_params},
                                 {'gaussian': periodic_params},
                                 nlv=0.02, freq_res=0.5)
    # ====================================================================
    # 2. RUN SPECPARAM EXTRACTION
    # ====================================================================

    print("Fitting Specparam models...")
    fm_off = SpectralModel(aperiodic_mode='knee')
    fm_on = SpectralModel(aperiodic_mode='knee')
    # Feed the pre-calculated PSD directly into Specparam
    fm_off.fit(freqs, psd_off, fre_range)
    fm_on.fit(freqs, psd_on, fre_range)

    # Extract Aperiodic Params (Using scikit-learn style attributes)
    ap_off = fm_off.get_params('aperiodic')
    ap_on = fm_on.get_params('aperiodic')

    # Aperiodic signal
    apfit_off = ap_off[0] - np.log10(freqs**ap_off[1])
    apfit_on = ap_on[0] - np.log10(freqs**ap_on[1])

    # Extract Flattened Spectra
    flat_spectra_off = fm_off.data.power_spectrum - apfit_off
    flat_spectra_on = fm_on.data.power_spectrum - apfit_on

    def extract_band_power(freq_array, spectra_array, fmin, fmax):
        idx = np.logical_and(freq_array >= fmin, freq_array <= fmax)
        return np.mean(spectra_array[idx])

    results = {
        'Condition': ['Simulated_StimOFF', 'Simulated_StimON'],
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
    print("\n--- Specparam & Power Results Table ---")
    print(df_results.to_markdown(index=False))

    # ====================================================================
    # 3. PLOTTING
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(freqs, psd_off, label='Stim OFF', color='blue')
    axes[0].loglog(freqs, psd_on, label='Stim ON', color='red')
    axes[0].axvspan(1, 12, color='gray', alpha=0.2, label='1-12 Hz')
    axes[0].axvspan(12, 35, color='orange', alpha=0.2, label='12-35 Hz')
    axes[0].set_title('Raw Power Spectrum (sim_power_spectrum)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power (V^2/Hz)')
    axes[0].legend()

    axes[1].plot(freqs, flat_spectra_off, label='Stim OFF', color='blue')
    axes[1].plot(freqs, flat_spectra_on, label='Stim ON', color='red')
    axes[1].axvspan(1, 12, color='gray', alpha=0.2)
    axes[1].axvspan(12, 35, color='orange', alpha=0.2)
    axes[1].set_title('Specparam Adjusted Periodic Power')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Log Power (Aperiodic Removed)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_pipeline_with_specparam_sim()