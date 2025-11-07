# =========================
# GW150914 Q-Transform Analysis Script
# =========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Parameters ----
EVENT_GPS = 1126259462
DURATION = 32  # seconds (±16s)
START_TIME = EVENT_GPS - DURATION // 2
END_TIME = EVENT_GPS + DURATION // 2
FREQUENCY_RANGE = (20, 500)  # Hz
Q_VALUE = 8  # Typical for BBH signals

# Output file names
H1_DATA_FILE = "GW150914_H1_strain.hdf5"
L1_DATA_FILE = "GW150914_L1_strain.hdf5"
H1_QSCAN_FILE = "GW150914_H1_qscan.hdf5"
L1_QSCAN_FILE = "GW150914_L1_qscan.hdf5"
QSCAN_PLOT_FILE = "GW150914_qtransform_spectrograms.png"

# ---- Task 1: Download Strain Data ----
print("="*60)
print("Task 1: Downloading GW150914 strain data for H1 and L1 detectors")
print("="*60)

strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {START_TIME} to {END_TIME}...")
    strain_H1 = TimeSeries.fetch_open_data('H1', START_TIME, END_TIME, cache=True)
    print("H1 strain data successfully downloaded.")
    # Save to file
    strain_H1.write(H1_DATA_FILE, overwrite=True)
    print(f"H1 strain data saved to {H1_DATA_FILE}")
except Exception as e:
    print(f"Error fetching or saving H1 data: {e}")

try:
    print(f"Fetching L1 strain data from {START_TIME} to {END_TIME}...")
    strain_L1 = TimeSeries.fetch_open_data('L1', START_TIME, END_TIME, cache=True)
    print("L1 strain data successfully downloaded.")
    # Save to file
    strain_L1.write(L1_DATA_FILE, overwrite=True)
    print(f"L1 strain data saved to {L1_DATA_FILE}")
except Exception as e:
    print(f"Error fetching or saving L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("Critical error: Could not fetch both H1 and L1 data. Exiting.")
    exit(1)

# ---- Task 2: Q-Transform Analysis ----
print("\n" + "="*60)
print("Task 2: Computing Q-transform for H1 and L1 strain data")
print("="*60)

qscan_H1 = None
qscan_L1 = None

try:
    print("Computing Q-transform for H1 strain data...")
    qscan_H1 = strain_H1.q_transform(frange=FREQUENCY_RANGE, qrange=(Q_VALUE, Q_VALUE))
    print("Q-transform for H1 completed.")
    # Save Q-transform to file
    qscan_H1.write(H1_QSCAN_FILE, overwrite=True)
    print(f"H1 Q-transform saved to {H1_QSCAN_FILE}")
except Exception as e:
    print(f"Error computing or saving Q-transform for H1: {e}")

try:
    print("Computing Q-transform for L1 strain data...")
    qscan_L1 = strain_L1.q_transform(frange=FREQUENCY_RANGE, qrange=(Q_VALUE, Q_VALUE))
    print("Q-transform for L1 completed.")
    # Save Q-transform to file
    qscan_L1.write(L1_QSCAN_FILE, overwrite=True)
    print(f"L1 Q-transform saved to {L1_QSCAN_FILE}")
except Exception as e:
    print(f"Error computing or saving Q-transform for L1: {e}")

if qscan_H1 is None or qscan_L1 is None:
    print("Critical error: Could not compute both H1 and L1 Q-transforms. Exiting.")
    exit(1)

# ---- Task 3: Visualization ----
print("\n" + "="*60)
print("Task 3: Plotting Q-transform spectrograms for H1 and L1")
print("="*60)

def plot_qtransform(qscan, ax, title):
    try:
        # Extract the spectrogram data and normalize
        energy = np.abs(qscan.value)
        norm_energy = energy / np.max(energy)
        
        # Prepare time and frequency axes
        times = qscan.times.value
        freqs = qscan.frequencies.value
        
        # Plot using imshow for full control
        im = ax.imshow(
            norm_energy.T,
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap='viridis'
        )
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('Normalized Energy')
    except Exception as e:
        print(f"Error plotting Q-transform for {title}: {e}")

try:
    print("Generating Q-transform spectrogram plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plot_qtransform(qscan_H1, axes[0], 'H1 Q-transform')
    plot_qtransform(qscan_L1, axes[1], 'L1 Q-transform')
    plt.tight_layout()
    plt.savefig(QSCAN_PLOT_FILE)
    plt.show()
    print(f"Q-transform spectrograms plotted and saved to {QSCAN_PLOT_FILE}")
except Exception as e:
    print(f"Error during visualization: {e}")

print("\nAll tasks completed successfully.")