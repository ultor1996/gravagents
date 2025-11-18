# =========================
# GW150914 Strain Data Download and Visualization
# =========================

# ---- Imports ----
from pycbc.catalog import Merger
from pycbc.frame import query_and_read_frame
import numpy as np
import matplotlib.pyplot as plt
import sys

# ---- Parameters ----
EVENT_NAME = "GW150914"
DETECTORS = ['H1', 'L1']
DURATION = 4  # seconds
GPS_CENTER = 1126259462
START_TIME = GPS_CENTER - DURATION // 2
END_TIME = GPS_CENTER + DURATION // 2
PLOT_FILENAME = "GW150914_strain_vs_time.png"

# ---- Task 1: Download and Load Strain Data ----
print("="*60)
print(f"Step 1: Downloading strain data for {EVENT_NAME} from LOSC")
print("="*60)

strain_data = {}

try:
    print("Querying event information from PyCBC catalog...")
    event = Merger(EVENT_NAME)
    print(f"Event found: {event.event}")

    for det in DETECTORS:
        print(f"Downloading strain data for detector {det}...")
        # Channel name is the same as detector
        try:
            strain = query_and_read_frame(
                det,
                det,  # Channel name is 'H1' or 'L1'
                START_TIME,
                END_TIME
            )
            strain_data[det] = strain
            print(f"  - {det}: Loaded {strain.duration} seconds at {strain.sample_rate} Hz")
        except Exception as det_e:
            print(f"  - Error loading data for {det}: {det_e}")

    if not strain_data:
        print("No strain data was loaded for any detector. Exiting.")
        sys.exit(1)

except Exception as e:
    print(f"An error occurred while downloading or loading the strain data: {e}")
    sys.exit(1)

# ---- Task 2: Plot Strain vs Time ----
print("\n" + "="*60)
print("Step 2: Plotting strain vs time for GW150914")
print("="*60)

try:
    plt.figure(figsize=(10, 6))
    colors = {'H1': 'tab:blue', 'L1': 'tab:orange'}
    plotted_any = False

    for det in DETECTORS:
        if det in strain_data:
            strain = strain_data[det]
            rel_time = strain.sample_times - GPS_CENTER
            plt.plot(rel_time, strain, label=f"{det}", color=colors[det])
            print(f"  - Plotted strain data for {det}.")
            plotted_any = True
        else:
            print(f"  - Warning: No strain data found for {det}.")

    if not plotted_any:
        print("No data available to plot. Exiting.")
        sys.exit(1)

    plt.xlabel("Time (seconds relative to GW150914)")
    plt.ylabel("Strain")
    plt.title("GW150914 Strain vs Time (Hanford & Livingston)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)
    print(f"Plot saved as '{PLOT_FILENAME}'.")
    plt.show()
    print("Plotting complete.")

except Exception as e:
    print(f"An error occurred during plotting: {e}")
    sys.exit(1)