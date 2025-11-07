# =========================
# GW150914 Strain Data Analysis: Download, Filter, and Plot
# =========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

# ---- Parameters ----
EVENT_TIME = 1126259462  # GW150914 event GPS time
HALF_WINDOW = 4096 // 2
START_TIME = EVENT_TIME - HALF_WINDOW
END_TIME = EVENT_TIME + HALF_WINDOW
DETECTORS = ['H1', 'L1']
RAW_DATA_DIR = "gw_data_raw"
FILTERED_DATA_DIR = "gw_data_filtered"
PLOT_FILENAME = "GW150914_filtered_strain.png"

# ---- Utility Functions ----
def save_timeseries_to_hdf5(timeseries, filename):
    """Save a GWpy TimeSeries to HDF5."""
    if timeseries is None:
        print(f"Warning: No data to save for {filename}.")
        return
    try:
        timeseries.write(filename, format='hdf5', overwrite=True)
        print(f"Saved TimeSeries to {filename}")
    except Exception as e:
        print(f"Error saving TimeSeries to {filename}: {e}")

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# =========================
# 1. Download Strain Data
# =========================
print("="*40)
print("Step 1: Downloading strain data for GW150914")
print("="*40)

ensure_dir_exists(RAW_DATA_DIR)
strain_data = {}

for det in DETECTORS:
    print(f"\nAttempting to fetch data for {det} from {START_TIME} to {END_TIME}...")
    try:
        ts = TimeSeries.fetch_open_data(det, START_TIME, END_TIME, cache=True)
        if ts is None or len(ts) == 0:
            print(f"Warning: No data returned for {det}.")
            strain_data[det] = None
        else:
            print(f"Successfully fetched data for {det}.")
            # Save raw data
            raw_filename = os.path.join(RAW_DATA_DIR, f"{det}_GW150914_raw.hdf5")
            save_timeseries_to_hdf5(ts, raw_filename)
            strain_data[det] = ts
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

strain_H1 = strain_data['H1']
strain_L1 = strain_data['L1']

# =========================
# 2. Filter Strain Data
# =========================
print("\n" + "="*40)
print("Step 2: Filtering strain data (30 Hz highpass, 250 Hz lowpass)")
print("="*40)

ensure_dir_exists(FILTERED_DATA_DIR)
filtered_strain = {}

for det, strain in [('H1', strain_H1), ('L1', strain_L1)]:
    print(f"\nProcessing {det} strain data...")
    try:
        if strain is None or len(strain) == 0:
            print(f"Warning: No data to filter for {det}. Skipping.")
            filtered_strain[det] = None
            continue
        # Apply highpass filter at 30 Hz
        print(f"Applying highpass filter at 30 Hz to {det}...")
        strain_hp = strain.highpass(30)
        # Apply lowpass filter at 250 Hz
        print(f"Applying lowpass filter at 250 Hz to {det}...")
        strain_bp = strain_hp.lowpass(250)
        filtered_strain[det] = strain_bp
        print(f"Filtering complete for {det}.")
        # Save filtered data
        filtered_filename = os.path.join(FILTERED_DATA_DIR, f"{det}_GW150914_filtered.hdf5")
        save_timeseries_to_hdf5(strain_bp, filtered_filename)
    except Exception as e:
        print(f"Error filtering data for {det}: {e}")
        filtered_strain[det] = None

filtered_strain_H1 = filtered_strain['H1']
filtered_strain_L1 = filtered_strain['L1']

# =========================
# 3. Plot Filtered Strain Data
# =========================
print("\n" + "="*40)
print("Step 3: Plotting filtered strain data")
print("="*40)

if filtered_strain_H1 is None or filtered_strain_L1 is None:
    print("Error: Filtered strain data not available for one or both detectors. Cannot plot.")
else:
    print("Preparing to plot filtered strain data for H1 and L1...")

    # Convert GPS times to seconds relative to event for better visualization
    t_H1 = filtered_strain_H1.times.value - EVENT_TIME
    t_L1 = filtered_strain_L1.times.value - EVENT_TIME

    plt.figure(figsize=(12, 6))
    plt.plot(t_H1, filtered_strain_H1.value, label='H1', color='C0', linewidth=0.8)
    plt.plot(t_L1, filtered_strain_L1.value, label='L1', color='C1', linewidth=0.8, alpha=0.8)
    plt.axvline(0, color='k', linestyle='--', label='GW150914 Event Time')
    plt.xlabel('Time (s) relative to GW150914')
    plt.ylabel('Strain')
    plt.title('Filtered Strain vs Time for GW150914 (H1 and L1)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    print(f"Saving plot to {PLOT_FILENAME}...")
    plt.savefig(PLOT_FILENAME, dpi=150)
    print("Displaying plot...")
    plt.show()

print("\nWorkflow complete.")