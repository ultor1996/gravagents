# =========================
# GW170817 Time-Frequency Analysis Integrated Script
# =========================

# ---- Imports ----
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries as PyCBC_TimeSeries  # Only needed if using PyCBC-specific methods
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
import h5py

# ---- Constants ----
GW170817_GPS = 1187008882.43
RAW_WINDOW_SEC = 0.5  # For 1-second window centered on merger
PLOT_WINDOW_SEC = 1.0  # For 2-second plot window
QTRANS_WINDOW_SEC = 0.5  # For 1-second Q-transform window
Q_FREQ_MIN = 30
Q_FREQ_MAX = 500
Q_VALUE = 8
DETECTORS = ['H1', 'L1']

# ---- Output Directories ----
OUTPUT_DIR = "gw170817_analysis_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =========================
# 1. Data Loading
# =========================
print("\n=== [1/4] DATA LOADING: Downloading raw strain data for GW170817 ===")
start_time = GW170817_GPS - RAW_WINDOW_SEC
end_time = GW170817_GPS + RAW_WINDOW_SEC
strain_data = {}

for det in DETECTORS:
    print(f"  Attempting to download strain data for {det} from {start_time} to {end_time}...")
    try:
        # Remove deprecated 'cache' argument per database guidance
        strain = GWpyTimeSeries.fetch_open_data(det, start_time, end_time)
        strain_data[det] = strain
        print(f"    Successfully downloaded strain data for {det}.")
        # Save raw data to HDF5 for reproducibility
        raw_file = os.path.join(OUTPUT_DIR, f"{det}_raw_strain.hdf5")
        strain.write(raw_file, format='hdf5')
        print(f"    Raw strain data for {det} saved to {raw_file}.")
    except Exception as e:
        print(f"    Error downloading data for {det}: {e}")
        strain_data[det] = None

# =========================
# 2. Preprocessing
# =========================
print("\n=== [2/4] PREPROCESSING: Whitening and bandpass filtering ===")
preprocessed_data = {}

for det in DETECTORS:
    print(f"\n  Preprocessing {det} data...")
    try:
        strain = strain_data[det]
        if strain is None:
            print(f"    No data available for {det}, skipping.")
            preprocessed_data[det] = None
            continue

        # Whitening
        print(f"    Whitening {det} data...")
        whitened = strain.whiten(4, 2)

        # Highpass filter at 30 Hz
        print(f"    Applying highpass filter at 30 Hz to {det} data...")
        highpassed = whitened.highpass(30.0, filtfilt=True)

        # Lowpass filter at 250 Hz
        print(f"    Applying lowpass filter at 250 Hz to {det} data...")
        bandpassed = highpassed.lowpass(250.0, filtfilt=True)

        preprocessed_data[det] = bandpassed
        print(f"    {det} data preprocessing complete.")

        # Save preprocessed data to HDF5
        pre_file = os.path.join(OUTPUT_DIR, f"{det}_preprocessed_strain.hdf5")
        bandpassed.write(pre_file, format='hdf5')
        print(f"    Preprocessed strain data for {det} saved to {pre_file}.")

    except Exception as e:
        print(f"    Error preprocessing {det} data: {e}")
        preprocessed_data[det] = None

# =========================
# 3. Time-Domain Visualization
# =========================
print("\n=== [3/4] TIME-DOMAIN VISUALIZATION: Plotting processed strain vs. time ===")
plot_start = GW170817_GPS - PLOT_WINDOW_SEC
plot_end = GW170817_GPS + PLOT_WINDOW_SEC

try:
    print("  Preparing data for visualization...")

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False

    for det, color in zip(DETECTORS, ['C0', 'C1']):
        strain = preprocessed_data.get(det)
        if strain is None:
            print(f"    Warning: No preprocessed data for {det}, skipping plot.")
            continue

        # Select the 2-second window centered on the merger
        mask = (strain.times.value >= plot_start) & (strain.times.value <= plot_end)
        times = strain.times.value[mask] - GW170817_GPS  # relative to merger time
        data = strain.value[mask]

        if len(times) == 0:
            print(f"    Warning: No data in plot window for {det}, skipping.")
            continue

        ax.plot(times, data, label=f'{det}', color=color, alpha=0.8)
        plotted_any = True

    # Mark the merger time (t=0)
    ax.axvline(0, color='k', linestyle='--', label='Merger Time')

    ax.set_xlabel('Time (s) relative to merger')
    ax.set_ylabel('Whitened, Bandpassed Strain')
    ax.set_title('GW170817: Whitened & Bandpassed Strain vs. Time (H1 & L1)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Save and show plot
    time_plot_file = os.path.join(OUTPUT_DIR, "strain_vs_time.png")
    if plotted_any:
        plt.savefig(time_plot_file)
        plt.show()
        print(f"  Time-domain plot saved to {time_plot_file}.")
    else:
        print("  No data available for time-domain plot.")

except Exception as e:
    print(f"  Error during visualization: {e}")

# =========================
# 4. Q-Transform Spectrograms
# =========================
print("\n=== [4/4] Q-TRANSFORM SPECTROGRAMS: Generating high-resolution spectrograms ===")

for det in DETECTORS:
    print(f"\n  Generating Q-transform spectrogram for {det}...")
    try:
        strain = preprocessed_data.get(det)
        if strain is None:
            print(f"    No preprocessed data for {det}, skipping.")
            continue

        # Select 1-second window centered on merger
        mask = (strain.times.value >= GW170817_GPS - QTRANS_WINDOW_SEC) & (strain.times.value <= GW170817_GPS + QTRANS_WINDOW_SEC)
        times = strain.times.value[mask]
        data = strain.value[mask]

        if len(data) == 0:
            print(f"    No data in selected window for {det}, skipping.")
            continue

        # Convert to GWpy TimeSeries for Q-transform
        gwpy_strain = GWpyTimeSeries(data, sample_rate=strain.sample_rate.value, t0=times[0])

        # Compute Q-transform
        print(f"    Computing Q-transform for {det}...")
        qspec = gwpy_strain.q_transform(
            outseg=(GW170817_GPS - QTRANS_WINDOW_SEC, GW170817_GPS + QTRANS_WINDOW_SEC),
            qrange=(Q_VALUE, Q_VALUE),
            frange=(Q_FREQ_MIN, Q_FREQ_MAX),
            logf=True
        )

        # Plot
        print(f"    Plotting Q-transform for {det}...")
        fig = qspec.plot(figsize=(10, 5), vmin=0.1, vmax=1.0)
        ax = fig.gca()
        # Mark merger time
        ax.axvline(GW170817_GPS, color='r', linestyle='--', label='Merger Time')
        ax.set_title(f'{det} Q-transform Spectrogram (GW170817)')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('GPS Time [s]')
        ax.legend()
        plt.tight_layout()

        # Save and show plot
        q_plot_file = os.path.join(OUTPUT_DIR, f"{det}_qtransform.png")
        plt.savefig(q_plot_file)
        plt.show()
        print(f"    Q-transform spectrogram for {det} saved to {q_plot_file}.")

    except Exception as e:
        print(f"    Error generating Q-transform for {det}: {e}")

print("\n=== Analysis Complete! All outputs saved in:", OUTPUT_DIR, "===\n")