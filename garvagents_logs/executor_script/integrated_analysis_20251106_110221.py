# --- Imports ---
import os
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
event_gps = 1187008882.4
window = 64  # seconds before and after event for data download
start = event_gps - window
end = event_gps + window
detectors = ['H1', 'L1']

zoom_window = 2  # seconds before and after event for plotting
plot_start = event_gps - zoom_window
plot_end = event_gps + zoom_window

output_dir = "gw170817_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download Strain Data ---
print("="*60)
print("TASK 1: Downloading strain data for GW170817 (H1 and L1)")
strain_data = {}

for det in detectors:
    try:
        print(f"Fetching strain data for {det} from GWOSC: {start} to {end} (GPS)...")
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        strain_data[det] = ts
        # Save raw data
        raw_path = os.path.join(output_dir, f"{det}_raw_strain.hdf5")
        ts.write(raw_path, overwrite=True)
        print(f"Successfully downloaded and saved raw data for {det}.")
    except Exception as e:
        print(f"Could not fetch data for {det}: {e}")
        strain_data[det] = None

# --- Task 2: Filtering ---
print("="*60)
print("TASK 2: Filtering strain data (bandpass 30-500 Hz)")
filtered_strain_data = {}

for det, ts in strain_data.items():
    if ts is None:
        print(f"Skipping {det}: No strain data available.")
        filtered_strain_data[det] = None
        continue
    try:
        print(f"Applying bandpass filter to {det} (30-500 Hz)...")
        ts_filtered = ts.highpass(30).lowpass(500)
        filtered_strain_data[det] = ts_filtered
        # Save filtered data
        filtered_path = os.path.join(output_dir, f"{det}_filtered_strain.hdf5")
        ts_filtered.write(filtered_path, overwrite=True)
        print(f"Filtering complete and saved for {det}.")
    except Exception as e:
        print(f"Error filtering {det}: {e}")
        filtered_strain_data[det] = None

# --- Task 3: Time Series Plot ---
print("="*60)
print("TASK 3: Plotting filtered strain time series around GW170817")
for det, ts in filtered_strain_data.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available.")
        continue
    try:
        print(f"Plotting filtered strain time series for {det}...")
        # Crop to zoom window
        ts_zoom = ts.crop(plot_start, plot_end)
        fig = ts_zoom.plot(figsize=(12, 4))
        ax = fig.gca()
        ax.set_title(f"{det} Filtered Strain around GW170817")
        ax.set_xlabel("Time [s, GPS]")
        ax.set_ylabel("Strain")
        ax.axvline(event_gps, color='r', linestyle='--', label='GW170817')
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{det}_filtered_strain_timeseries.png")
        plt.savefig(plot_path)
        print(f"Time series plot for {det} saved to {plot_path}")
        plt.show()
    except Exception as e:
        print(f"Error plotting time series for {det}: {e}")

# --- Task 4: Q-transform Spectrogram ---
print("="*60)
print("TASK 4: Generating and plotting Q-transform spectrograms")
for det, ts in filtered_strain_data.items():
    if ts is None:
        print(f"Skipping {det}: No filtered data available.")
        continue
    try:
        print(f"Generating Q-transform spectrogram for {det}...")
        # Crop to zoom window
        ts_zoom = ts.crop(plot_start, plot_end)
        # Perform Q-transform
        q = ts_zoom.q_transform(outseg=(plot_start, plot_end))
        fig = q.plot(figsize=(12, 6), vmin=1e-24, vmax=1e-21)
        ax = fig.gca()
        ax.set_title(f"{det} Q-transform Spectrogram around GW170817")
        ax.set_xlabel("Time [s, GPS]")
        ax.set_ylabel("Frequency [Hz]")
        ax.axvline(event_gps, color='r', linestyle='--', label='GW170817')
        ax.legend()
        plt.tight_layout()
        q_plot_path = os.path.join(output_dir, f"{det}_qtransform.png")
        plt.savefig(q_plot_path)
        print(f"Q-transform plot for {det} saved to {q_plot_path}")
        plt.show()
    except Exception as e:
        print(f"Error generating Q-transform for {det}: {e}")

print("="*60)
print("Analysis complete. All outputs saved in:", output_dir)