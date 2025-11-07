# GW170817 Time-Frequency Analysis: Data Retrieval, Preprocessing, Visualization, and Q-transform
# ------------------------------------------------------------------------------

# === Imports ===
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
gps_merger = 1187008882.43
data_duration = 1.0  # seconds for data retrieval and Q-transform
plot_window = 2.0    # seconds for strain visualization
qrange = (8, 64)
frange = (30, 250)

# Channels for H1 and L1
channels = {
    'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
    'L1': 'L1:GWOSC-4KHZ_R1_STRAIN'
}

# ------------------------------------------------------------------------------
# Task 1: Data Loading
# ------------------------------------------------------------------------------
print("\n=== Task 1: Downloading raw strain data for GW170817 ===")
start_time = gps_merger - data_duration / 2
end_time = gps_merger + data_duration / 2

strain_data = {}
for det, channel in channels.items():
    print(f"Attempting to download strain data for {det} from {start_time} to {end_time} (GPS)...")
    try:
        ts = TimeSeries.get(channel, start_time, end_time, cache=True)
        strain_data[det] = ts
        print(f"Successfully downloaded {det} strain data.")
    except Exception as e:
        print(f"Error downloading {det} strain data: {e}")
        strain_data[det] = None

# ------------------------------------------------------------------------------
# Task 2: Preprocessing (Whitening and Bandpass Filtering)
# ------------------------------------------------------------------------------
print("\n=== Task 2: Preprocessing strain data (whitening and bandpass filtering) ===")
preprocessed_data = {}

for det in ['H1', 'L1']:
    print(f"\nPreprocessing {det} strain data...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"Warning: No data found for {det}. Skipping preprocessing.")
        preprocessed_data[det] = None
        continue
    try:
        print(f"  Whitening {det} data...")
        ts_white = ts.whiten()
        print(f"  Applying high-pass filter at 30 Hz to {det} data...")
        ts_hp = ts_white.highpass(30)
        print(f"  Applying low-pass filter at 250 Hz to {det} data...")
        ts_bp = ts_hp.lowpass(250)
        preprocessed_data[det] = ts_bp
        print(f"  {det} data preprocessing complete.")
    except Exception as e:
        print(f"Error preprocessing {det} data: {e}")
        preprocessed_data[det] = None

# ------------------------------------------------------------------------------
# Task 3: Strain Visualization
# ------------------------------------------------------------------------------
print("\n=== Task 3: Visualizing processed strain for both detectors ===")
plot_start = gps_merger - plot_window / 2
plot_end = gps_merger + plot_window / 2

plt.figure(figsize=(10, 6))
colors = {'H1': 'C0', 'L1': 'C1'}
plotted = False

for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"Warning: No preprocessed data for {det}. Skipping plot.")
        continue
    try:
        ts_window = ts.crop(plot_start, plot_end)
        t = ts_window.times.value - gps_merger
        plt.plot(t, ts_window.value, label=f'{det}', color=colors[det])
        plotted = True
        print(f"Plotted {det} strain data.")
    except Exception as e:
        print(f"Error plotting {det} data: {e}")

if not plotted:
    print("No data was plotted. Exiting visualization.")
else:
    plt.axvline(0, color='k', linestyle='--', label='Merger (t=0)')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened, bandpassed strain')
    plt.title('GW170817: Whitened & Bandpassed Strain (H1 & L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# Task 4: Q-transform Spectrograms
# ------------------------------------------------------------------------------
print("\n=== Task 4: Generating Q-transform spectrograms ===")
q_start = gps_merger - data_duration / 2
q_end = gps_merger + data_duration / 2

for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"Warning: No preprocessed data for {det}. Skipping Q-transform.")
        continue
    try:
        print(f"\nGenerating Q-transform for {det}...")
        ts_window = ts.crop(q_start, q_end)
        qspec = ts_window.q_transform(qrange=qrange, frange=frange)
        print(f"Q-transform computed for {det}. Plotting spectrogram...")
        fig = qspec.plot(figsize=(10, 5), vmin=0.0001, vmax=0.01, cmap='viridis')
        ax = fig.gca()
        ax.set_title(f'{det} Q-transform Spectrogram (GW170817)')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s] relative to merger')
        ax.set_xlim(q_start - gps_merger, q_end - gps_merger)
        ax.axvline(0, color='r', linestyle='--', label='Merger (t=0)')
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating or plotting Q-transform for {det}: {e}")

print("\n=== Analysis complete. ===")