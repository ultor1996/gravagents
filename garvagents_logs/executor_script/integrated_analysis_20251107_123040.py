# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parameters ---
gw_event_gps = 1126259462
window = 16  # seconds before and after event
start_time = gw_event_gps - window
end_time = gw_event_gps + window
detectors = ['H1', 'L1']
qrange = (4, 64)
frange = (20, 512)  # Hz

# Create output directory
output_dir = "gw150914_qtransform_results"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download GW150914 strain data for H1 and L1 detectors ---
print("="*60)
print("TASK 1: Downloading GW150914 strain data for H1 and L1 detectors")
print("="*60)
strain_data = {}

for det in detectors:
    print(f"\nFetching strain data for {det} from {start_time} to {end_time} (GPS)...")
    try:
        # Remove deprecated cache argument per GWpy API change
        ts = TimeSeries.fetch_open_data(det, start_time, end_time)
        strain_data[det] = ts
        # Save strain data to file for reproducibility
        ts.write(os.path.join(output_dir, f"{det}_strain_gw150914.gwf"), format='gwf')
        print(f"Successfully fetched and saved data for {det}.")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

# Save strain_data keys for reference
np.savez_compressed(os.path.join(output_dir, "strain_data_keys.npz"),
                    detectors=detectors, available=[d for d in detectors if strain_data[d] is not None])

# --- Task 2: Compute Q-transform for both H1 and L1 strain data ---
print("\n" + "="*60)
print("TASK 2: Computing Q-transform for H1 and L1")
print("="*60)
qtransforms = {}

for det in detectors:
    print(f"\nComputing Q-transform for {det}...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"No strain data available for {det}, skipping Q-transform.")
        qtransforms[det] = None
        continue
    try:
        q = ts.q_transform(qrange=qrange, frange=frange)
        # Check if q.value, q.xindex.value, q.yindex.value are non-empty
        if (q.value.size == 0 or q.xindex.value.size == 0 or q.yindex.value.size == 0):
            print(f"Q-transform for {det} is empty, skipping.")
            qtransforms[det] = None
            continue
        qtransforms[det] = q
        # Save Q-transform as a numpy array for reference (not full GWpy object)
        np.savez_compressed(os.path.join(output_dir, f"{det}_qtransform.npz"),
                            data=q.value, times=q.xindex.value, freqs=q.yindex.value)
        print(f"Q-transform for {det} computed and saved successfully.")
    except Exception as e:
        print(f"Error computing Q-transform for {det}: {e}")
        qtransforms[det] = None

# --- Task 3: Plot Q-transform spectrograms with normalized energy colorbar ---
print("\n" + "="*60)
print("TASK 3: Plotting Q-transform spectrograms for H1 and L1")
print("="*60)

for det in detectors:
    print(f"\nPlotting Q-transform spectrogram for {det}...")
    q = qtransforms.get(det)
    if q is None:
        print(f"No Q-transform data available for {det}, skipping plot.")
        continue
    try:
        fig = q.plot(norm='log', vmin=1e-24, vmax=1e-21)
        ax = fig.gca()
        ax.set_title(f"{det} Q-transform Spectrogram (GW150914)")
        ax.set_ylabel("Frequency [Hz]")
        # Defensive: check q.xindex.value is not empty before using
        if hasattr(q, 'xindex') and hasattr(q.xindex, 'value') and len(q.xindex.value) > 0:
            xlabel_time = int(q.xindex.value[0])
        else:
            xlabel_time = start_time
        ax.set_xlabel(f"Time [s] (GPS {xlabel_time})")
        # Ensure colorbar is labeled, but check ax.images is not empty
        if hasattr(ax, 'images') and len(ax.images) > 0:
            cbar = fig.colorbar(ax.images[0], ax=ax)
            cbar.set_label("Normalized energy")
        else:
            print(f"Warning: No image found in axes for {det}, skipping colorbar.")
        plt.tight_layout()
        # Save plot to file
        plot_path = os.path.join(output_dir, f"{det}_qtransform_spectrogram.png")
        fig.savefig(plot_path, dpi=150)
        plt.show()
        print(f"Spectrogram for {det} displayed and saved to {plot_path}.")
    except Exception as e:
        print(f"Error plotting Q-transform for {det}: {e}")

print("\nAll tasks completed. Results saved in:", output_dir)