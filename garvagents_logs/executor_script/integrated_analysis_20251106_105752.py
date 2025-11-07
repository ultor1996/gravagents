# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

import os

# --- Parameters ---
gps_center = 1180922494.5
duration = 64  # seconds
gps_start = gps_center - duration / 2
gps_end = gps_center + duration / 2
detectors = ['H1', 'L1']

# Output directories
output_dir = "gw170608_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download Strain Data ---
print("="*60)
print("TASK 1: Downloading strain data for GW170608 (H1 and L1)")
strain_data = {}

for det in detectors:
    print(f"Fetching strain data for {det} from {gps_start} to {gps_end}...")
    try:
        # CRITICAL: Remove 'cache' argument per GWpy API change
        ts = TimeSeries.fetch_open_data(det, gps_start, gps_end)
        print(f"Successfully fetched data for {det}.")
        # Save raw data to disk
        raw_path = os.path.join(output_dir, f"{det}_raw_strain.hdf5")
        ts.write(raw_path, overwrite=True)
        print(f"Raw strain data for {det} saved to {raw_path}")
        strain_data[det] = ts
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

# Data integrity and sampling rate check
try:
    if strain_data['H1'] is not None and strain_data['L1'] is not None:
        h1_rate = strain_data['H1'].sample_rate.value
        l1_rate = strain_data['L1'].sample_rate.value
        print(f"H1 sample rate: {h1_rate} Hz")
        print(f"L1 sample rate: {l1_rate} Hz")
        if not np.isclose(h1_rate, l1_rate):
            raise ValueError("Sampling rates for H1 and L1 do not match!")
        else:
            print("Sampling rates match.")
    else:
        print("One or both detectors failed to load data.")
except Exception as e:
    print(f"Error during data integrity check: {e}")

# --- Task 2: Filtering ---
print("="*60)
print("TASK 2: Filtering strain data (highpass 30 Hz, lowpass 200 Hz)")
filtered_strain_data = {}

for det in detectors:
    ts = strain_data.get(det)
    if ts is None:
        print(f"Skipping {det}: No data available.")
        filtered_strain_data[det] = None
        continue
    try:
        print(f"Applying highpass filter at 30 Hz to {det}...")
        ts_hp = ts.highpass(30)
        print(f"Applying lowpass filter at 200 Hz to {det}...")
        ts_bp = ts_hp.lowpass(200)
        filtered_strain_data[det] = ts_bp
        # Save filtered data to disk
        filtered_path = os.path.join(output_dir, f"{det}_filtered_strain.hdf5")
        ts_bp.write(filtered_path, overwrite=True)
        print(f"Filtered strain data for {det} saved to {filtered_path}")
        print(f"Filtering complete for {det}.")
    except Exception as e:
        print(f"Error filtering {det}: {e}")
        filtered_strain_data[det] = None

# --- Task 3: Time Series Plot ---
print("="*60)
print("TASK 3: Plotting filtered strain vs time for both detectors")
plt.figure(figsize=(12, 6))
colors = {'H1': 'tab:blue', 'L1': 'tab:orange'}
plotted = False

for det in detectors:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"Skipping {det}: No filtered data available.")
        continue
    try:
        plt.plot(ts.times.value, ts.value, label=det, color=colors[det], linewidth=1)
        plotted = True
        print(f"Plotted filtered strain for {det}.")
    except Exception as e:
        print(f"Error plotting {det}: {e}")

if plotted:
    plt.xlabel("Time (s, GPS)")
    plt.ylabel("Strain")
    plt.title("Filtered Strain vs Time for H1 and L1 Detectors")
    plt.legend()
    plt.tight_layout()
    timeseries_plot_path = os.path.join(output_dir, "filtered_strain_timeseries.png")
    plt.savefig(timeseries_plot_path)
    print(f"Time series plot saved to {timeseries_plot_path}")
    plt.show()
else:
    print("No data was plotted. Please check the filtered_strain_data dictionary.")

# --- Task 4: Q-transform Spectrograms ---
print("="*60)
print("TASK 4: Generating and plotting Q-transform spectrograms")
for det in detectors:
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"Skipping {det}: No filtered data available.")
        continue
    try:
        print(f"Computing Q-transform for {det}...")
        # Use correct time attributes for outseg
        t0 = ts.t0.value
        t1 = t0 + ts.duration.value
        q = ts.q_transform(outseg=(t0, t1))
        print(f"Plotting Q-transform for {det}...")
        fig = q.plot(figsize=(12, 6))
        ax = fig.gca()
        ax.set_title(f"Q-transform Spectrogram for {det}")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s, GPS]")
        plt.tight_layout()
        q_plot_path = os.path.join(output_dir, f"{det}_qtransform.png")
        plt.savefig(q_plot_path)
        print(f"Q-transform plot for {det} saved to {q_plot_path}")
        plt.show()
    except Exception as e:
        print(f"Error generating or plotting Q-transform for {det}: {e}")

print("="*60)
print("Analysis complete. All outputs saved in:", output_dir)