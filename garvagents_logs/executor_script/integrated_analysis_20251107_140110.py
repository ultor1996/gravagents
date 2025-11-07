# =========================
# GW150914 Whitening and Filtering Script
# =========================

# ---- Imports ----
import os
import traceback

from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# ---- Section 1: Data Loading ----
print("="*40)
print("Step 1: Downloading GW150914 strain data for H1 and L1")
print("="*40)

gps_center = 1126259462.4
duration = 4  # seconds
start = gps_center - duration / 2
end = gps_center + duration / 2

detectors = ['H1', 'L1']
strain_data = {}

for det in detectors:
    try:
        print(f"Attempting to download strain data for {det} from {start} to {end}...")
        strain = TimeSeries.fetch_open_data(det, start, end)
        strain_data[det] = strain
        print(f"Successfully downloaded strain data for {det}.")
        # Save strain data to HDF5 for reproducibility
        out_fname = f"GW150914_{det}_strain.hdf5"
        print(f"Saving {det} strain data to {out_fname}...")
        strain.write(out_fname, format='hdf5')
    except Exception as e:
        print(f"Failed to download strain data for {det}: {e}")
        traceback.print_exc()
        strain_data[det] = None

# ---- Section 2: Whitening ----
print("\n" + "="*40)
print("Step 2: Whitening strain data")
print("="*40)

whitened_strain = {}

for det in detectors:
    strain = strain_data.get(det)
    if strain is None:
        print(f"No strain data available for {det}, skipping whitening.")
        whitened_strain[det] = None
        continue
    try:
        print(f"Whitening strain data for {det}...")
        whitened = strain.whiten()
        whitened_strain[det] = whitened
        print(f"Strain data for {det} whitened successfully.")
        # Save whitened data to HDF5
        out_fname = f"GW150914_{det}_whitened.hdf5"
        print(f"Saving {det} whitened data to {out_fname}...")
        whitened.write(out_fname, format='hdf5')
    except Exception as e:
        print(f"Error whitening strain data for {det}: {e}")
        traceback.print_exc()
        whitened_strain[det] = None

# ---- Section 3: Filtering ----
print("\n" + "="*40)
print("Step 3: Filtering whitened data (30-250 Hz)")
print("="*40)

filtered_strain = {}

for det in detectors:
    whitened = whitened_strain.get(det)
    if whitened is None:
        print(f"No whitened data available for {det}, skipping filtering.")
        filtered_strain[det] = None
        continue
    try:
        print(f"Applying highpass filter at 30 Hz to {det} data...")
        hp = whitened.highpass(30)
        print(f"Applying lowpass filter at 250 Hz to {det} data...")
        bp = hp.lowpass(250)
        filtered_strain[det] = bp
        print(f"Filtering complete for {det}.")
        # Save filtered data to HDF5
        out_fname = f"GW150914_{det}_filtered.hdf5"
        print(f"Saving {det} filtered data to {out_fname}...")
        bp.write(out_fname, format='hdf5')
    except Exception as e:
        print(f"Error filtering data for {det}: {e}")
        traceback.print_exc()
        filtered_strain[det] = None

# ---- Section 4: Visualization ----
print("\n" + "="*40)
print("Step 4: Plotting whitened and filtered strain data")
print("="*40)

plt.figure(figsize=(10, 6))
colors = {'H1': 'tab:blue', 'L1': 'tab:orange'}
plotted = False
t0 = None

for det in detectors:
    ts = filtered_strain.get(det)
    if ts is None:
        print(f"No filtered data for {det}, skipping plot.")
        continue
    try:
        print(f"Plotting data for {det}...")
        plt.plot(ts.times.value, ts.value, label=det, color=colors[det])
        plotted = True
        if t0 is None:
            t0 = ts.t0.value
    except Exception as e:
        print(f"Error plotting data for {det}: {e}")
        traceback.print_exc()

if plotted:
    plt.xlabel('Time (s) since GPS {:.1f}'.format(t0 if t0 is not None else gps_center))
    plt.ylabel('Whitened, filtered strain')
    plt.title('Whitened and Filtered Strain Data for GW150914 (H1 & L1)')
    plt.legend()
    plt.tight_layout()
    plot_fname = "GW150914_whitened_filtered_strain.png"
    try:
        print(f"Saving plot to {plot_fname}...")
        plt.savefig(plot_fname)
    except Exception as e:
        print(f"Error saving plot: {e}")
        traceback.print_exc()
    plt.show()
else:
    print("No data was plotted. Please check previous processing steps.")

print("\nAll steps completed.")