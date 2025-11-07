# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# --- Parameters and Setup ---
merger_time = 1126259462.43
duration = 4  # seconds
half_duration = duration / 2
start_time = merger_time - half_duration
end_time = merger_time + half_duration
detectors = ['H1', 'L1']

# --- Task 1: Data Loading ---
print("="*60)
print("Step 1: Downloading strain data for GW150914 (H1 and L1)")
print("="*60)
strain_data = {}

for det in detectors:
    print(f"Fetching data for {det} from {start_time} to {end_time} (GPS)...")
    try:
        ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        strain_data[det] = ts
        print(f"Successfully fetched data for {det}.")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

# --- Task 2: Filtering ---
print("\n" + "="*60)
print("Step 2: Applying bandpass filter (30-250 Hz)")
print("="*60)
filtered_strain_data = {}

for det in detectors:
    print(f"Applying bandpass filter (30-250 Hz) to {det} data...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"No data available for {det}, skipping filtering.")
        filtered_strain_data[det] = None
        continue
    try:
        ts_filtered = ts.highpass(30).lowpass(250)
        filtered_strain_data[det] = ts_filtered
        print(f"Filtering complete for {det}.")
        # Save filtered data for reproducibility
        np.save(f"filtered_strain_{det}.npy", ts_filtered.value)
    except Exception as e:
        print(f"Error filtering data for {det}: {e}")
        filtered_strain_data[det] = None

# --- Task 3: Whitening/Normalization ---
print("\n" + "="*60)
print("Step 3: Whitening or normalizing filtered data")
print("="*60)
whitened_strain_data = {}

for det in detectors:
    print(f"Whitening (or normalizing) filtered data for {det}...")
    ts = filtered_strain_data.get(det)
    if ts is None:
        print(f"No filtered data available for {det}, skipping whitening/normalization.")
        whitened_strain_data[det] = None
        continue
    try:
        ts_whitened = ts.whiten()
        whitened_strain_data[det] = ts_whitened
        print(f"Whitening complete for {det}.")
        np.save(f"whitened_strain_{det}.npy", ts_whitened.value)
    except Exception as e:
        print(f"Whitening failed for {det} with error: {e}")
        print("Attempting normalization by standard deviation instead...")
        try:
            ts_normalized = ts / np.std(ts.value)
            whitened_strain_data[det] = ts_normalized
            print(f"Normalization complete for {det}.")
            np.save(f"normalized_strain_{det}.npy", ts_normalized.value)
        except Exception as e2:
            print(f"Normalization also failed for {det}: {e2}")
            whitened_strain_data[det] = None

# --- Task 4: Visualization ---
print("\n" + "="*60)
print("Step 4: Plotting filtered and whitened/normalized strain")
print("="*60)

plt.figure(figsize=(10, 6))
colors = {'H1': 'C0', 'L1': 'C1'}
labels = {'H1': 'H1', 'L1': 'L1'}
data_plotted = False

for det in detectors:
    ts = whitened_strain_data.get(det)
    if ts is None:
        print(f"No whitened/normalized data for {det}, skipping plot.")
        continue
    # Time relative to merger
    time_rel = ts.times.value - merger_time
    plt.plot(time_rel, ts.value, label=labels[det], color=colors[det], alpha=0.8)
    data_plotted = True

if not data_plotted:
    print("No data available to plot. Exiting visualization.")
else:
    plt.axvline(0, color='k', linestyle='--', label='Merger Time (t=0)')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Strain (whitened/normalized)')
    plt.title('GW150914: Filtered and Whitened/Normalized Strain (H1 & L1)')
    plt.legend()
    plt.xlim(-0.2, 0.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW150914_strain_plot.png")
    print("Plot saved as 'GW150914_strain_plot.png'.")
    plt.show()

print("\nWorkflow complete.")