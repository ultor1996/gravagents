# ============================================================
# GW150914 Strain Data Download, Filtering, and Visualization
# ============================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.types import TimeSeries

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------
event_name = "GW150914"
event_time = 1126259462.4  # GPS time of GW150914
segment_duration = 4  # seconds
segment_start = event_time - segment_duration / 2
segment_end = event_time + segment_duration / 2

low_freq = 35.0
high_freq = 350.0

# Output file names
output_prefix = "GW150914"
output_files = {
    "strain_H1": f"{output_prefix}_strain_H1.npy",
    "strain_L1": f"{output_prefix}_strain_L1.npy",
    "strain_times_H1": f"{output_prefix}_strain_times_H1.npy",
    "strain_times_L1": f"{output_prefix}_strain_times_L1.npy",
    "filtered_strain_H1": f"{output_prefix}_filtered_strain_H1.npy",
    "filtered_strain_L1": f"{output_prefix}_filtered_strain_L1.npy",
    "plot": f"{output_prefix}_filtered_strain_plot.png"
}

# -------------------------------
# TASK 1: DATA LOADING
# -------------------------------
print("="*60)
print("TASK 1: Downloading GW150914 strain data from LIGO Open Science Center")
print("="*60)

try:
    print(f"Fetching event '{event_name}' from PyCBC catalog...")
    merger = Merger(event_name)
    print("Event found. Downloading strain data...")

    # Download strain for Hanford (H1)
    print("Downloading Hanford (H1) strain data...")
    strain_H1 = merger.strain('H1', segment_start, segment_end)
    strain_times_H1 = np.linspace(segment_start, segment_end, len(strain_H1))
    print(f"Hanford strain data downloaded: {len(strain_H1)} samples.")

    # Download strain for Livingston (L1)
    print("Downloading Livingston (L1) strain data...")
    strain_L1 = merger.strain('L1', segment_start, segment_end)
    strain_times_L1 = np.linspace(segment_start, segment_end, len(strain_L1))
    print(f"Livingston strain data downloaded: {len(strain_L1)} samples.")

    # Save raw data for reproducibility
    np.save(output_files["strain_H1"], strain_H1)
    np.save(output_files["strain_L1"], strain_L1)
    np.save(output_files["strain_times_H1"], strain_times_H1)
    np.save(output_files["strain_times_L1"], strain_times_L1)
    print("Raw strain data and time arrays saved to disk.")

except Exception as e:
    print(f"Error occurred during data download: {e}")
    sys.exit(1)

# -------------------------------
# TASK 2: DATA PROCESSING
# -------------------------------
print("\n" + "="*60)
print("TASK 2: Bandpass filtering of strain data (35–350 Hz)")
print("="*60)

try:
    print("Calculating sampling rate from Hanford data...")
    dt_H1 = strain_times_H1[1] - strain_times_H1[0]
    sample_rate_H1 = 1.0 / dt_H1
    print(f"Hanford sample rate: {sample_rate_H1:.2f} Hz")

    print("Calculating sampling rate from Livingston data...")
    dt_L1 = strain_times_L1[1] - strain_times_L1[0]
    sample_rate_L1 = 1.0 / dt_L1
    print(f"Livingston sample rate: {sample_rate_L1:.2f} Hz")

    print("Applying bandpass filter to Hanford (H1) data...")
    ts_H1 = TimeSeries(strain_H1, delta_t=dt_H1, epoch=strain_times_H1[0])
    filtered_H1 = ts_H1.bandpass(low_freq, high_freq)
    print("Hanford data filtered.")

    print("Applying bandpass filter to Livingston (L1) data...")
    ts_L1 = TimeSeries(strain_L1, delta_t=dt_L1, epoch=strain_times_L1[0])
    filtered_L1 = ts_L1.bandpass(low_freq, high_freq)
    print("Livingston data filtered.")

    # Convert filtered data to numpy arrays for plotting
    filtered_strain_H1 = filtered_H1.numpy()
    filtered_strain_L1 = filtered_L1.numpy()

    # Save filtered data
    np.save(output_files["filtered_strain_H1"], filtered_strain_H1)
    np.save(output_files["filtered_strain_L1"], filtered_strain_L1)
    print("Filtered strain data saved to disk.")

except Exception as e:
    print(f"Error during bandpass filtering: {e}")
    sys.exit(1)

# -------------------------------
# TASK 3: VISUALIZATION
# -------------------------------
print("\n" + "="*60)
print("TASK 3: Plotting filtered strain vs time for both detectors")
print("="*60)

try:
    print("Preparing time arrays relative to event time...")
    time_rel_H1 = strain_times_H1 - event_time
    time_rel_L1 = strain_times_L1 - event_time

    print("Creating the plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(time_rel_H1, filtered_strain_H1, label='Hanford (H1)', color='C0')
    plt.plot(time_rel_L1, filtered_strain_L1, label='Livingston (L1)', color='C1')
    plt.xlabel('Time (s) relative to GW150914')
    plt.ylabel('Strain (filtered)')
    plt.title('Filtered Strain vs Time for GW150914')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_files["plot"])
    plt.show()
    print(f"Plot displayed and saved as '{output_files['plot']}'.")

except Exception as e:
    print(f"Error during plotting: {e}")
    sys.exit(1)

print("\nAll tasks completed successfully.")