# =========================
# GW170817 Time-Frequency Analysis Pipeline
# =========================

# ---- Imports ----
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from gwpy.timeseries import TimeSeries

# ---- Parameters ----
gps_center = 1187008882.43
duration_data = 1.0  # seconds for raw data retrieval
window_vis = 2.0     # seconds for visualization and Q-transform
start_time_data = gps_center - duration_data / 2
end_time_data = gps_center + duration_data / 2
start_time_vis = gps_center - window_vis / 2
end_time_vis = gps_center + window_vis / 2

# ---- Task 1: Data Loading ----
print("="*40)
print("Task 1: Loading GW170817 strain data")
print("="*40)
strain_H1 = None
strain_L1 = None
try:
    print("Loading GW170817 event from PyCBC catalog...")
    merger = Merger("GW170817")
    print("Successfully loaded GW170817 event.")

    print(f"Retrieving strain data for Hanford (H1)...")
    # Retrieve full strain, then slice to desired window
    strain_H1_full = merger.strain('H1')
    strain_H1 = strain_H1_full.time_slice(start_time_data, end_time_data)
    print("Successfully retrieved H1 strain data.")

    print(f"Retrieving strain data for Livingston (L1)...")
    strain_L1_full = merger.strain('L1')
    strain_L1 = strain_L1_full.time_slice(start_time_data, end_time_data)
    print("Successfully retrieved L1 strain data.")

except Exception as e:
    print(f"Error occurred during data loading: {e}")
    raise

# ---- Task 2: Preprocessing (Whitening & Bandpass) ----
print("\n" + "="*40)
print("Task 2: Preprocessing strain data (whitening and bandpass filtering)")
print("="*40)
strain_H1_whitened_filtered = None
strain_L1_whitened_filtered = None
try:
    assert strain_H1 is not None and strain_L1 is not None

    print("Whitening Hanford (H1) strain data...")
    strain_H1_white = strain_H1.whiten()
    print("Applying 30 Hz highpass filter to H1...")
    strain_H1_hp = strain_H1_white.highpass(30.0)
    print("Applying 250 Hz lowpass filter to H1...")
    strain_H1_whitened_filtered = strain_H1_hp.lowpass(250.0)
    print("H1 preprocessing complete.")

    print("Whitening Livingston (L1) strain data...")
    strain_L1_white = strain_L1.whiten()
    print("Applying 30 Hz highpass filter to L1...")
    strain_L1_hp = strain_L1_white.highpass(30.0)
    print("Applying 250 Hz lowpass filter to L1...")
    strain_L1_whitened_filtered = strain_L1_hp.lowpass(250.0)
    print("L1 preprocessing complete.")

except AssertionError:
    print("Strain data for H1 and/or L1 not loaded. Please run the data loading step first.")
    raise
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# ---- Task 3: Strain Visualization ----
print("\n" + "="*40)
print("Task 3: Visualizing processed strain data")
print("="*40)
try:
    assert strain_H1_whitened_filtered is not None and strain_L1_whitened_filtered is not None

    print("Extracting 2-second window around merger for H1...")
    h1_window = strain_H1_whitened_filtered.time_slice(start_time_vis, end_time_vis)
    print("Extracting 2-second window around merger for L1...")
    l1_window = strain_L1_whitened_filtered.time_slice(start_time_vis, end_time_vis)

    # Prepare time arrays relative to merger time
    h1_times = h1_window.sample_times - gps_center
    l1_times = l1_window.sample_times - gps_center

    print("Plotting strain data for both detectors...")
    plt.figure(figsize=(10, 6))
    plt.plot(h1_times, h1_window, label='H1 (Hanford)', color='C0')
    plt.plot(l1_times, l1_window, label='L1 (Livingston)', color='C1', alpha=0.7)
    plt.axvline(0, color='k', linestyle='--', label='Merger Time (t=0)')
    plt.axvspan(-0.1, 0.1, color='red', alpha=0.2, label='Transient Signal')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened, bandpassed strain')
    plt.title('GW170817: Whitened & Bandpassed Strain (H1 & L1)\n2-second window centered on merger')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW170817_strain_whitened_bandpassed.png", dpi=150)
    plt.show()
    print("Visualization complete. Figure saved as 'GW170817_strain_whitened_bandpassed.png'.")

except AssertionError:
    print("Preprocessed strain data not found. Please run preprocessing first.")
    raise
except Exception as e:
    print(f"Error during visualization: {e}")
    raise

# ---- Task 4: Q-transform Spectrogram ----
print("\n" + "="*40)
print("Task 4: Q-transform spectrograms")
print("="*40)
try:
    assert strain_H1_whitened_filtered is not None and strain_L1_whitened_filtered is not None

    # Helper: Convert PyCBC TimeSeries to GWpy TimeSeries
    def pycbc_to_gwpy(pycbc_ts):
        return TimeSeries(pycbc_ts.numpy(), 
                          sample_rate=pycbc_ts.sample_rate, 
                          t0=pycbc_ts.start_time)

    print("Converting H1 and L1 data to GWpy TimeSeries...")
    h1_gwpy = pycbc_to_gwpy(strain_H1_whitened_filtered)
    l1_gwpy = pycbc_to_gwpy(strain_L1_whitened_filtered)

    print("Extracting 2-second window for Q-transform...")
    h1_window_gwpy = h1_gwpy.crop(start_time_vis, end_time_vis)
    l1_window_gwpy = l1_gwpy.crop(start_time_vis, end_time_vis)

    # Q-transform parameters
    qrange = (4, 64)
    frange = (30, 250)
    print("Computing Q-transform for H1...")
    h1_q = h1_window_gwpy.q_transform(outseg=(start_time_vis, end_time_vis), qrange=qrange, frange=frange)
    print("Computing Q-transform for L1...")
    l1_q = l1_window_gwpy.q_transform(outseg=(start_time_vis, end_time_vis), qrange=qrange, frange=frange)

    # Plotting
    print("Plotting Q-transform spectrograms...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # H1
    ax = axs[0]
    h1_q.plot(ax=ax, cmap='viridis', vmin=0, vmax=5, norm='log')
    ax.axvline(gps_center, color='r', linestyle='--', label='Merger Time')
    ax.set_title('GW170817 H1 Q-transform Spectrogram')
    ax.set_ylabel('Frequency [Hz]')
    ax.legend()

    # L1
    ax = axs[1]
    l1_q.plot(ax=ax, cmap='viridis', vmin=0, vmax=5, norm='log')
    ax.axvline(gps_center, color='r', linestyle='--', label='Merger Time')
    ax.set_title('GW170817 L1 Q-transform Spectrogram')
    ax.set_xlabel('GPS Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.legend()

    plt.tight_layout()
    plt.savefig("GW170817_Qtransform_spectrograms.png", dpi=150)
    plt.show()
    print("Q-transform visualization complete. Figure saved as 'GW170817_Qtransform_spectrograms.png'.")

except AssertionError:
    print("Preprocessed strain data not found. Please run preprocessing first.")
    raise
except Exception as e:
    print(f"Error during Q-transform spectrogram generation: {e}")
    raise

print("\nAll tasks completed successfully.")