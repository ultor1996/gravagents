# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Section 1: Data Downloading ---
print("="*60)
print("Step 1: Downloading GW150914 strain data for H1 and L1 detectors")
print("="*60)

# Define GPS time and window
gps_merger = 1126259462.4
window = 2  # seconds before and after
start = gps_merger - window
end = gps_merger + window

# Prepare output variables
strain_H1 = None
strain_L1 = None

print(f"Attempting to download strain data for H1 and L1 from {start} to {end} (GPS seconds)...")

try:
    print("Fetching H1 strain data...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start, end, cache=True)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error fetching H1 data: {e}")
    strain_H1 = None

try:
    print("Fetching L1 strain data...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start, end, cache=True)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"Error fetching L1 data: {e}")
    strain_L1 = None

if strain_H1 is None and strain_L1 is None:
    print("ERROR: Failed to download strain data for both detectors. Exiting.")
    sys.exit(1)

# --- Section 2: Filtering ---
print("\n" + "="*60)
print("Step 2: Filtering strain data (highpass 30 Hz, lowpass 250 Hz)")
print("="*60)

filtered_H1 = None
filtered_L1 = None

def filter_strain(strain, label):
    if strain is None:
        print(f"Warning: No data found for {label}. Skipping filtering.")
        return None
    try:
        print(f"Applying highpass filter at 30 Hz to {label}...")
        hp = strain.highpass(30)
        print(f"Applying lowpass filter at 250 Hz to {label}...")
        lp = hp.lowpass(250)
        print(f"Filtering complete for {label}.")
        return lp
    except Exception as e:
        print(f"Error filtering {label}: {e}")
        return None

filtered_H1 = filter_strain(strain_H1, "H1")
filtered_L1 = filter_strain(strain_L1, "L1")

if filtered_H1 is None and filtered_L1 is None:
    print("ERROR: Filtering failed for both detectors. Exiting.")
    sys.exit(1)

# --- Section 3: Save Filtered Data ---
print("\n" + "="*60)
print("Step 3: Saving filtered data to disk")
print("="*60)

output_dir = "filtered_data"
os.makedirs(output_dir, exist_ok=True)

def save_timeseries(ts, filename):
    if ts is not None:
        try:
            ts.write(filename, format='hdf5')
            print(f"Saved filtered data to {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

save_timeseries(filtered_H1, os.path.join(output_dir, "filtered_H1.hdf5"))
save_timeseries(filtered_L1, os.path.join(output_dir, "filtered_L1.hdf5"))

# --- Section 4: Visualization ---
print("\n" + "="*60)
print("Step 4: Plotting filtered strain data")
print("="*60)

def plot_filtered_strain(filtered_H1, filtered_L1, gps_merger):
    if filtered_H1 is None and filtered_L1 is None:
        print("No filtered data available for either detector. Cannot plot.")
        return
    print("Preparing to plot filtered strain data...")

    plt.figure(figsize=(12, 6))

    # Plot H1
    if filtered_H1 is not None:
        plt.plot(filtered_H1.times.value, filtered_H1.value, label='H1', color='C0')
    else:
        print("Warning: No filtered H1 data to plot.")

    # Plot L1
    if filtered_L1 is not None:
        plt.plot(filtered_L1.times.value, filtered_L1.value, label='L1', color='C1')
    else:
        print("Warning: No filtered L1 data to plot.")

    # Mark the merger time
    plt.axvline(gps_merger, color='k', linestyle='--', label='Merger Time')

    # Labels and legend
    plt.xlabel('GPS Time (s)')
    plt.ylabel('Strain')
    plt.title('Filtered Strain vs Time for H1 and L1 Detectors')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    print("Displaying plot...")
    plt.show()

plot_filtered_strain(filtered_H1, filtered_L1, gps_merger)

print("\nWorkflow complete.")