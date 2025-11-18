# =========================
# GW150914 Q-Transform Analysis Script
# =========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ---- Parameters ----
GW_EVENT_GPS = 1126259462
PAD = 16  # seconds before and after event
START_TIME = GW_EVENT_GPS - PAD
END_TIME = GW_EVENT_GPS + PAD
DETECTORS = ['H1', 'L1']

Q_VALUE = 8
FMIN = 20
FMAX = 500

# Output file names
STRAIN_DATA_FILE = "gw150914_strain_data.npz"
QTRANSFORMS_FILE = "gw150914_qtransforms.npz"
PLOT_FILE = "gw150914_qtransform_spectrograms.png"

# ---- Task 1: Download Strain Data ----
print("="*60)
print("Step 1: Downloading GW150914 strain data for H1 and L1 detectors...")
strain_data = {}

for det in DETECTORS:
    print(f"Fetching strain data for {det} from {START_TIME} to {END_TIME} (GPS)...")
    try:
        ts = TimeSeries.fetch_open_data(
            det, START_TIME, END_TIME, 
            quality='DATA'
        )
        strain_data[det] = ts
        print(f"  Successfully fetched data for {det}.")
    except Exception as e:
        print(f"  ERROR: Could not fetch data for {det}: {e}")
        strain_data[det] = None

# Save strain data for reproducibility (as numpy arrays)
try:
    np.savez(
        STRAIN_DATA_FILE,
        H1=strain_data['H1'].value if strain_data['H1'] is not None else np.array([]),
        L1=strain_data['L1'].value if strain_data['L1'] is not None else np.array([]),
        H1_times=strain_data['H1'].times.value if strain_data['H1'] is not None else np.array([]),
        L1_times=strain_data['L1'].times.value if strain_data['L1'] is not None else np.array([])
    )
    print(f"Strain data saved to '{STRAIN_DATA_FILE}'.")
except Exception as e:
    print(f"WARNING: Could not save strain data: {e}")

# ---- Task 2: Compute Q-Transforms ----
print("="*60)
print("Step 2: Computing Q-transforms for H1 and L1...")

qtransforms = {}

for det in DETECTORS:
    print(f"Computing Q-transform for {det}...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"  No strain data available for {det}, skipping Q-transform.")
        qtransforms[det] = None
        continue
    try:
        q = ts.q_transform(
            qrange=(Q_VALUE, Q_VALUE),
            frange=(FMIN, FMAX)
        )
        abs_q = q.abs()
        max_abs_q = np.max(abs_q)
        if max_abs_q == 0:
            print(f"  WARNING: Max Q-transform energy is zero for {det}.")
            normalized_energy = abs_q
        else:
            normalized_energy = abs_q / max_abs_q
        qtransforms[det] = {
            'qtransform': q,
            'normalized_energy': normalized_energy
        }
        print(f"  Q-transform computed and normalized for {det}.")
    except Exception as e:
        print(f"  ERROR: Could not compute Q-transform for {det}: {e}")
        qtransforms[det] = None

# Save Q-transform normalized energies for reproducibility
try:
    np.savez(
        QTRANSFORMS_FILE,
        H1_energy=qtransforms['H1']['normalized_energy'] if qtransforms['H1'] is not None else np.array([]),
        L1_energy=qtransforms['L1']['normalized_energy'] if qtransforms['L1'] is not None else np.array([]),
    )
    print(f"Q-transform normalized energies saved to '{QTRANSFORMS_FILE}'.")
except Exception as e:
    print(f"WARNING: Could not save Q-transform data: {e}")

# ---- Task 3: Visualization ----
print("="*60)
print("Step 3: Plotting normalized Q-transform spectrograms...")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
mesh = None  # For colorbar

for idx, det in enumerate(DETECTORS):
    print(f"Plotting Q-transform for {det}...")
    qinfo = qtransforms.get(det)
    if qinfo is None or qinfo['qtransform'] is None:
        print(f"  No Q-transform data for {det}, skipping plot.")
        axes[idx].set_visible(False)
        continue

    q = qinfo['qtransform']
    norm_energy = qinfo['normalized_energy']

    # Get time and frequency axes
    times = q.xindex.value  # seconds (relative to GPS start)
    freqs = q.yindex.value  # Hz

    # Compute absolute GPS times for x-axis
    gps_start = q.epoch.value
    abs_times = times + gps_start

    # Plot spectrogram
    mesh = axes[idx].pcolormesh(
        abs_times, freqs, norm_energy.T,  # .T to match axes
        shading='auto', cmap='viridis', vmin=0, vmax=1
    )
    axes[idx].axvline(GW_EVENT_GPS, color='r', linestyle='--', label='GW150914')
    axes[idx].set_ylabel('Frequency [Hz]')
    axes[idx].set_title(f'{det} Q-transform (Normalized)')
    axes[idx].legend(loc='upper right')

# Colorbar (shared)
if mesh is not None:
    cbar = fig.colorbar(mesh, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Normalized energy')

axes[-1].set_xlabel('GPS Time [s]')
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.show()
print(f"Q-transform spectrograms plotted and saved to '{PLOT_FILE}'.")
print("="*60)
print("Analysis complete.")