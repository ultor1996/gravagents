# ==========================
# GW150914 Strain Data Download and Visualization
# ==========================

# ---- Imports ----
from pycbc.catalog import Merger
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ---- Configuration ----
EVENT_NAME = "GW150914"
WINDOW_DURATION = 4  # seconds
OUTPUT_DIR = "gw150914_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Task 1: Download Strain Data ----
print("="*60)
print(f"Step 1: Downloading strain data for event {EVENT_NAME}")
print("="*60)

try:
    print(f"Fetching event information for {EVENT_NAME}...")
    event = Merger(EVENT_NAME)
    event_time = event.time
    print(f"Event time (GPS): {event_time}")

    start_time = event_time - WINDOW_DURATION / 2
    end_time = event_time + WINDOW_DURATION / 2

    # Download Hanford (H1) strain data
    print("Downloading Hanford (H1) strain data...")
    strain_H1 = event.strain('H1', start_time, end_time)
    print(f"Hanford (H1) strain data loaded: {len(strain_H1)} samples.")

    # Download Livingston (L1) strain data
    print("Downloading Livingston (L1) strain data...")
    strain_L1 = event.strain('L1', start_time, end_time)
    print(f"Livingston (L1) strain data loaded: {len(strain_L1)} samples.")

    # Save strain data for reproducibility
    np.save(os.path.join(OUTPUT_DIR, "strain_H1.npy"), strain_H1)
    np.save(os.path.join(OUTPUT_DIR, "strain_L1.npy"), strain_L1)
    with open(os.path.join(OUTPUT_DIR, "event_time.txt"), "w") as f:
        f.write(str(event_time))

    print(f"Strain data and event time saved in '{OUTPUT_DIR}'.")

except Exception as e:
    print(f"Error occurred while downloading strain data: {e}")
    print("Exiting script.")
    sys.exit(1)

# ---- Task 2: Plot Strain vs Time ----
print("\n" + "="*60)
print("Step 2: Plotting strain vs time for H1 and L1")
print("="*60)

try:
    print("Preparing time arrays for plotting...")
    # Get sample rate and start time from strain objects
    sr_H1 = strain_H1.sample_rate
    sr_L1 = strain_L1.sample_rate
    t0_H1 = strain_H1.start_time
    t0_L1 = strain_L1.start_time

    # Create time arrays relative to event time
    times_H1 = np.arange(len(strain_H1)) / sr_H1 + float(t0_H1) - float(event_time)
    times_L1 = np.arange(len(strain_L1)) / sr_L1 + float(t0_L1) - float(event_time)

    print("Plotting strain data for H1 and L1...")
    plt.figure(figsize=(12, 6))

    # Plot H1
    plt.subplot(2, 1, 1)
    plt.plot(times_H1, strain_H1, label='Hanford (H1)', color='C0')
    plt.axvline(0, color='k', linestyle='--', label='Event time')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Strain')
    plt.title('GW150914 Strain - Hanford (H1)')
    plt.legend()
    plt.grid(True)

    # Plot L1
    plt.subplot(2, 1, 2)
    plt.plot(times_L1, strain_L1, label='Livingston (L1)', color='C1')
    plt.axvline(0, color='k', linestyle='--', label='Event time')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Strain')
    plt.title('GW150914 Strain - Livingston (L1)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "GW150914_strain_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Plotting complete. Plot saved as '{plot_path}'.")

except Exception as e:
    print(f"Error during plotting: {e}")
    print("Exiting script.")
    sys.exit(2)

print("\nAll steps completed successfully.")