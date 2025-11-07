# =========================
# GW150914 H1 & L1 Strain Data Analysis
# =========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Configuration ----
event_gps = 1126259462.4
segment_duration = 4  # seconds
start_time = event_gps - segment_duration / 2
end_time = event_gps + segment_duration / 2

output_dir = "gw150914_results"
os.makedirs(output_dir, exist_ok=True)

# ---- 1. Download Strain Data ----
print("="*60)
print("STEP 1: Downloading H1 and L1 strain data for GW150914")
print("="*60)
strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {start_time} to {end_time} (GPS seconds)...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("Successfully fetched H1 strain data.")
except Exception as e:
    print(f"Error fetching H1 strain data: {e}")

try:
    print(f"Fetching L1 strain data from {start_time} to {end_time} (GPS seconds)...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("Successfully fetched L1 strain data.")
except Exception as e:
    print(f"Error fetching L1 strain data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("ERROR: Failed to load both H1 and L1 strain data. Exiting.")
    exit(1)
else:
    print("Both H1 and L1 strain data loaded successfully.")

# Save raw data for reproducibility
np.savetxt(os.path.join(output_dir, "strain_H1_raw.txt"),
           np.column_stack((strain_H1.times.value, strain_H1.value)),
           header="Time(s) Strain(H1)")
np.savetxt(os.path.join(output_dir, "strain_L1_raw.txt"),
           np.column_stack((strain_L1.times.value, strain_L1.value)),
           header="Time(s) Strain(L1)")

# ---- 2. Whiten the Data ----
print("\n" + "="*60)
print("STEP 2: Whitening the strain data")
print("="*60)
whitened_H1 = None
whitened_L1 = None

try:
    print("Whitening H1 strain data...")
    whitened_H1 = strain_H1.whiten()
    print("H1 strain data whitened successfully.")
except Exception as e:
    print(f"Error whitening H1 strain data: {e}")

try:
    print("Whitening L1 strain data...")
    whitened_L1 = strain_L1.whiten()
    print("L1 strain data whitened successfully.")
except Exception as e:
    print(f"Error whitening L1 strain data: {e}")

if whitened_H1 is None or whitened_L1 is None:
    print("ERROR: Whitening failed for one or both detectors. Exiting.")
    exit(1)

# Save whitened data
np.savetxt(os.path.join(output_dir, "strain_H1_whitened.txt"),
           np.column_stack((whitened_H1.times.value, whitened_H1.value)),
           header="Time(s) Whitened_Strain(H1)")
np.savetxt(os.path.join(output_dir, "strain_L1_whitened.txt"),
           np.column_stack((whitened_L1.times.value, whitened_L1.value)),
           header="Time(s) Whitened_Strain(L1)")

# ---- 3. Bandpass Filter ----
print("\n" + "="*60)
print("STEP 3: Applying bandpass filter (30-250 Hz)")
print("="*60)
filtered_H1 = None
filtered_L1 = None

try:
    print("Applying highpass (30 Hz) and lowpass (250 Hz) to H1...")
    hp_H1 = whitened_H1.highpass(30)
    filtered_H1 = hp_H1.lowpass(250)
    print("H1 data filtered successfully.")
except Exception as e:
    print(f"Error filtering H1 data: {e}")

try:
    print("Applying highpass (30 Hz) and lowpass (250 Hz) to L1...")
    hp_L1 = whitened_L1.highpass(30)
    filtered_L1 = hp_L1.lowpass(250)
    print("L1 data filtered successfully.")
except Exception as e:
    print(f"Error filtering L1 data: {e}")

if filtered_H1 is None or filtered_L1 is None:
    print("ERROR: Filtering failed for one or both detectors. Exiting.")
    exit(1)

# Save filtered data
np.savetxt(os.path.join(output_dir, "strain_H1_filtered.txt"),
           np.column_stack((filtered_H1.times.value, filtered_H1.value)),
           header="Time(s) Filtered_Strain(H1)")
np.savetxt(os.path.join(output_dir, "strain_L1_filtered.txt"),
           np.column_stack((filtered_L1.times.value, filtered_L1.value)),
           header="Time(s) Filtered_Strain(L1)")

# ---- 4. Plot the Results ----
print("\n" + "="*60)
print("STEP 4: Plotting filtered H1 and L1 strain data")
print("="*60)
try:
    print("Preparing to plot filtered H1 and L1 strain data...")

    # Extract time and strain arrays
    time_H1 = filtered_H1.times.value
    strain_H1_plot = filtered_H1.value
    time_L1 = filtered_L1.times.value
    strain_L1_plot = filtered_L1.value

    plt.figure(figsize=(10, 6))
    plt.plot(time_H1, strain_H1_plot, label='H1', color='blue', alpha=0.7)
    plt.plot(time_L1, strain_L1_plot, label='L1', color='orange', alpha=0.7)
    plt.xlabel('Time (s) since GPS {:.1f}'.format(filtered_H1.t0.value))
    plt.ylabel('Strain')
    plt.title('GW150914: Filtered Strain Data (H1 vs L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "GW150914_H1_L1_filtered_strain.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot displayed and saved to {plot_path}")

except Exception as e:
    print(f"Error during plotting: {e}")

print("\nAnalysis complete. All intermediate and final results saved in:", output_dir)