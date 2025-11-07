# =========================
# GW150914 Q-transform Analysis Script
# =========================

# ---- Imports ----
import os
import sys
import traceback

from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
import matplotlib.pyplot as plt
import h5py

# ---- Section 1: Data Loading ----
print("="*40)
print("Step 1: Downloading GW150914 strain data for H1 and L1")
print("="*40)

event_time = 1126259462.4
window = 2  # seconds before and after
start_time = event_time - window
end_time = event_time + window

detectors = ['H1', 'L1']
strain_data = {}

for det in detectors:
    try:
        print(f"Downloading strain data for {det} from {start_time} to {end_time}...")
        strain = TimeSeries.fetch_open_data(det, start_time, end_time)
        strain_data[det] = strain
        print(f"Successfully downloaded data for {det}.")
        # Save strain data to HDF5 for reproducibility
        out_fname = f"GW150914_{det}_strain.hdf5"
        print(f"Saving {det} strain data to {out_fname}...")
        strain.write(out_fname, format='hdf5')
    except Exception as e:
        print(f"Error downloading data for {det}: {e}")
        traceback.print_exc()
        strain_data[det] = None

# ---- Section 2: Q-transform Computation ----
print("\n" + "="*40)
print("Step 2: Computing Q-transform spectrograms")
print("="*40)

qrange = (8, 64)
frange = (20, 500)  # Hz
qtransforms = {}

for det in detectors:
    strain = strain_data.get(det)
    if strain is None:
        print(f"No strain data available for {det}, skipping Q-transform.")
        qtransforms[det] = None
        continue
    try:
        print(f"Computing Q-transform for {det}...")
        qspec = strain.q_transform(qrange=qrange, frange=frange)
        qtransforms[det] = qspec
        print(f"Q-transform for {det} computed successfully.")
        # Save Q-transform to HDF5
        out_fname = f"GW150914_{det}_qtransform.hdf5"
        print(f"Saving {det} Q-transform to {out_fname}...")
        qspec.write(out_fname, format='hdf5')
    except Exception as e:
        print(f"Error computing Q-transform for {det}: {e}")
        traceback.print_exc()
        qtransforms[det] = None

# ---- Section 3: Visualization ----
print("\n" + "="*40)
print("Step 3: Plotting Q-transform spectrograms")
print("="*40)

titles = {'H1': 'H1 Q-transform', 'L1': 'L1 Q-transform'}

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Q-transform Spectrograms for GW150914')

for idx, det in enumerate(detectors):
    qspec = qtransforms.get(det)
    ax = axes[idx]
    if qspec is None:
        print(f"No Q-transform data for {det}, skipping plot.")
        ax.set_visible(False)
        continue
    try:
        print(f"Plotting Q-transform for {det}...")
        # GWpy's plot returns (fig, ax, im)
        _, _, im = qspec.plot(ax=ax, vmin=0, vmax=1, norm='log')
        ax.set_title(titles[det])
        ax.set_ylabel('Frequency [Hz]')
        # Add colorbar for each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized energy')
    except Exception as e:
        print(f"Error plotting Q-transform for {det}: {e}")
        traceback.print_exc()
        ax.set_visible(False)

axes[-1].set_xlabel('Time [s]')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

# Save the figure
plot_fname = "GW150914_Qtransform_spectrograms.png"
try:
    print(f"Saving Q-transform spectrogram plot to {plot_fname}...")
    plt.savefig(plot_fname)
except Exception as e:
    print(f"Error saving plot: {e}")
    traceback.print_exc()

plt.show()

print("\nAll steps completed.")