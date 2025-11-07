# GW170817 Time-Frequency Analysis Integrated Script

# =========================
# Imports and Configuration
# =========================
import os
import sys
import traceback

from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Parameters and Paths
# =========================
# Analysis parameters
MERGER_GPS = 1187008882.4
RAW_WINDOW = 5        # seconds before/after merger for data download
PREPROC_LOW = 20      # Hz, bandpass low cutoff
PREPROC_HIGH = 250    # Hz, bandpass high cutoff
PLOT_WINDOW = 0.5     # seconds before/after merger for strain plot

# Q-transform parameters
QTRANSFORM_PARAMS = {
    'mismatch': 0.001,
    'logfsteps': 100,
    'qrange': (8, 8),
    'frange': (20, 512)
}

# Output directories
OUTPUT_DIR = "gw170817_analysis_outputs"
RAW_DATA_DIR = os.path.join(OUTPUT_DIR, "raw_data")
PREPROC_DATA_DIR = os.path.join(OUTPUT_DIR, "preprocessed_data")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for d in [OUTPUT_DIR, RAW_DATA_DIR, PREPROC_DATA_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================
# 1. Data Loading
# =========================
print("\n=== [1/4] Downloading GW170817 strain data (H1 & L1) ===")
start_time = MERGER_GPS - RAW_WINDOW
end_time = MERGER_GPS + RAW_WINDOW
detectors = {'H1': 'H1', 'L1': 'L1'}
strain_data = {}

for det, channel in detectors.items():
    print(f"  Attempting to download strain data for {det} from {start_time} to {end_time}...")
    try:
        ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        strain_data[det] = ts
        print(f"    Success: {det} data loaded. Duration: {ts.duration} s, Sample rate: {ts.sample_rate.value} Hz")
        # Save raw data to disk (GWpy TimeSeries can write to .txt or .hdf5)
        raw_path = os.path.join(RAW_DATA_DIR, f"{det}_raw.txt")
        ts.write(raw_path, format='txt')
        print(f"    Raw data saved to {raw_path}")
    except Exception as e:
        print(f"    ERROR: Could not download data for {det}: {e}")
        traceback.print_exc()
        strain_data[det] = None

# =========================
# 2. Preprocessing
# =========================
print("\n=== [2/4] Preprocessing strain data (whitening + bandpass) ===")
preprocessed_data = {}

for det in ['H1', 'L1']:
    ts = strain_data.get(det)
    if ts is None:
        print(f"  No raw data for {det}, skipping preprocessing.")
        preprocessed_data[det] = None
        continue
    try:
        print(f"  Preprocessing {det}...")
        # Whitening
        ts_white = ts.whiten()
        # Bandpass filter
        ts_white_bp = ts_white.bandpass(PREPROC_LOW, PREPROC_HIGH)
        preprocessed_data[det] = ts_white_bp
        print(f"    {det} preprocessing complete. Duration: {ts_white_bp.duration} s.")
        # Save preprocessed data
        preproc_path = os.path.join(PREPROC_DATA_DIR, f"{det}_whitened_bandpassed.txt")
        ts_white_bp.write(preproc_path, format='txt')
        print(f"    Preprocessed data saved to {preproc_path}")
    except Exception as e:
        print(f"    ERROR: Preprocessing failed for {det}: {e}")
        traceback.print_exc()
        preprocessed_data[det] = None

# =========================
# 3. Strain Visualization
# =========================
print("\n=== [3/4] Plotting preprocessed strain (±0.5s around merger) ===")
plot_start = MERGER_GPS - PLOT_WINDOW
plot_end = MERGER_GPS + PLOT_WINDOW
colors = {'H1': 'C0', 'L1': 'C1'}
labels = {'H1': 'Hanford (H1)', 'L1': 'Livingston (L1)'}
data_found = False

plt.figure(figsize=(10, 6))
for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"  No preprocessed data for {det}, skipping plot.")
        continue
    try:
        ts_zoom = ts.crop(plot_start, plot_end)
        times = ts_zoom.times.value - MERGER_GPS
        plt.plot(times, ts_zoom.value, label=labels[det], color=colors[det])
        data_found = True
        print(f"    Plotted {det}: {len(times)} samples in ±{PLOT_WINDOW} s window.")
    except Exception as e:
        print(f"    ERROR: Could not plot {det}: {e}")
        traceback.print_exc()

if data_found:
    plt.axvline(0, color='k', linestyle='--', label='Merger Time')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened, bandpassed strain')
    plt.title('GW170817: Whitened & Bandpassed Strain (±0.5 s around merger)')
    plt.legend()
    plt.tight_layout()
    strain_plot_path = os.path.join(PLOTS_DIR, "GW170817_strain_whitened_bandpassed.png")
    plt.savefig(strain_plot_path, dpi=150)
    print(f"  Strain plot saved to {strain_plot_path}")
    plt.show()
else:
    print("  No data available to plot.")

# =========================
# 4. Q-transform Spectrograms
# =========================
print("\n=== [4/4] Generating Q-transform spectrograms ===")
for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"  No preprocessed data for {det}, skipping Q-transform.")
        continue
    try:
        print(f"  Computing Q-transform for {det}...")
        q = ts.q_transform(**QTRANSFORM_PARAMS)
        print(f"    Q-transform computed. Plotting spectrogram...")
        fig = q.plot(figsize=(10, 6), vmin=0, vmax=10)
        ax = fig.gca()
        ax.set_title(f"{labels[det]} Q-transform Spectrogram (GW170817)")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel(f"Time [s] (GPS {ts.t0.value:.1f})")
        plt.tight_layout()
        qplot_path = os.path.join(PLOTS_DIR, f"{det}_Qtransform_spectrogram.png")
        plt.savefig(qplot_path, dpi=150)
        print(f"    Q-transform spectrogram saved to {qplot_path}")
        plt.show()
    except Exception as e:
        print(f"    ERROR: Q-transform failed for {det}: {e}")
        traceback.print_exc()

print("\n=== Analysis complete! ===")
print(f"All outputs saved in: {OUTPUT_DIR}")