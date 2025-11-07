# GW150914 Strain Data Download, Filtering, and Visualization
# -----------------------------------------------------------
# This script downloads the GW150914 strain data from the LIGO Open Science Center,
# applies a bandpass filter, and plots the filtered strain for both Hanford (H1)
# and Livingston (L1) detectors.

import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.filter import highpass, lowpass
import os

# -------------------------------
# SECTION 1: Data Loading
# -------------------------------
event_name = "GW150914"
window_duration = 4  # seconds

try:
    print(f"[1/3] Fetching event information for {event_name}...")
    merger = Merger(event_name)
    event_time = merger.time
    print(f"    Event GPS time: {event_time}")

    # Define start and end times for the window
    start_time = event_time - window_duration / 2
    end_time = event_time + window_duration / 2

    print("    Downloading strain data for Hanford (H1)...")
    strain_H1 = merger.strain('H1', start_time=start_time, end_time=end_time)
    print(f"    Hanford data loaded: {len(strain_H1)} samples.")

    print("    Downloading strain data for Livingston (L1)...")
    strain_L1 = merger.strain('L1', start_time=start_time, end_time=end_time)
    print(f"    Livingston data loaded: {len(strain_L1)} samples.")

    # Save results for later use
    gw150914_strain_H1 = strain_H1
    gw150914_strain_L1 = strain_L1
    gw150914_time = np.linspace(start_time, end_time, len(strain_H1))

    # Save raw data to disk
    np.save("gw150914_strain_H1_raw.npy", gw150914_strain_H1)
    np.save("gw150914_strain_L1_raw.npy", gw150914_strain_L1)
    np.save("gw150914_time.npy", gw150914_time)
    print("    Raw strain data and time array saved to disk.")

except Exception as e:
    print(f"[ERROR] Error occurred during data loading: {e}")
    sys.exit(1)

# -------------------------------
# SECTION 2: Data Processing (Filtering)
# -------------------------------
low_frequency = 35.0   # Hz
high_frequency = 350.0 # Hz

try:
    print("[2/3] Applying bandpass filter to Hanford (H1) data...")
    # Highpass first, then lowpass
    strain_H1_hp = highpass(gw150914_strain_H1, low_frequency)
    gw150914_strain_H1_filtered = lowpass(strain_H1_hp, high_frequency)
    print("    Hanford (H1) filtering complete.")

    print("    Applying bandpass filter to Livingston (L1) data...")
    strain_L1_hp = highpass(gw150914_strain_L1, low_frequency)
    gw150914_strain_L1_filtered = lowpass(strain_L1_hp, high_frequency)
    print("    Livingston (L1) filtering complete.")

    # Save filtered data to disk
    np.save("gw150914_strain_H1_filtered.npy", gw150914_strain_H1_filtered)
    np.save("gw150914_strain_L1_filtered.npy", gw150914_strain_L1_filtered)
    print("    Filtered strain data saved to disk.")

except Exception as e:
    print(f"[ERROR] Error during bandpass filtering: {e}")
    sys.exit(1)

# -------------------------------
# SECTION 3: Visualization
# -------------------------------
try:
    print("[3/3] Preparing to plot filtered strain data for both detectors...")

    # Calculate time relative to event for better visualization
    event_time_center = gw150914_time[0] + (gw150914_time[-1] - gw150914_time[0]) / 2
    time_relative = gw150914_time - event_time_center

    plt.figure(figsize=(12, 6))
    plt.plot(time_relative, gw150914_strain_H1_filtered, label='Hanford (H1)', color='C0')
    plt.plot(time_relative, gw150914_strain_L1_filtered, label='Livingston (L1)', color='C1')
    plt.axvline(0, color='k', linestyle='--', label='Event time')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Strain')
    plt.title('GW150914 Filtered Strain Data\nHanford (H1) and Livingston (L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gw150914_filtered_strain_plot.png", dpi=150)
    plt.show()

    print("    Plotting complete. Figure saved as 'gw150914_filtered_strain_plot.png'.")

except Exception as e:
    print(f"[ERROR] Error during plotting: {e}")
    sys.exit(1)