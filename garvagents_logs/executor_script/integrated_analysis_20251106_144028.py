# =========================
# GW150914 Q-transform Analysis Script
# =========================

# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parameters ---
GW_EVENT = "GW150914"
gw150914_gps_start = 1126259446
gw150914_duration = 32
qrange = (4, 64)
frange = (20, 512)  # Hz

# --- Output Paths ---
output_dir = "gw150914_outputs"
os.makedirs(output_dir, exist_ok=True)
strain_H1_path = os.path.join(output_dir, "strain_H1.hdf5")
strain_L1_path = os.path.join(output_dir, "strain_L1.hdf5")
qspec_H1_path = os.path.join(output_dir, "qspec_H1.hdf5")
qspec_L1_path = os.path.join(output_dir, "qspec_L1.hdf5")

# =========================
# 1. Download Strain Data
# =========================
print("="*40)
print("Step 1: Downloading strain data for GW150914")
print("="*40)

strain_H1 = None
strain_L1 = None

try:
    print("Fetching H1 strain data for GW150914...")
    strain_H1 = TimeSeries.fetch_open_data('H1', gw150914_gps_start, gw150914_gps_start + gw150914_duration)
    print("Successfully fetched H1 strain data.")
    # Save to disk
    strain_H1.write(strain_H1_path, overwrite=True)
    print(f"H1 strain data saved to {strain_H1_path}")
except Exception as e:
    print(f"Error fetching or saving H1 data: {e}")

try:
    print("Fetching L1 strain data for GW150914...")
    strain_L1 = TimeSeries.fetch_open_data('L1', gw150914_gps_start, gw150914_gps_start + gw150914_duration)
    print("Successfully fetched L1 strain data.")
    # Save to disk
    strain_L1.write(strain_L1_path, overwrite=True)
    print(f"L1 strain data saved to {strain_L1_path}")
except Exception as e:
    print(f"Error fetching or saving L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("Critical error: Could not fetch both H1 and L1 strain data. Exiting.")
    exit(1)

# =========================
# 2. Compute Q-transform
# =========================
print("\n" + "="*40)
print("Step 2: Computing Q-transform spectrograms")
print("="*40)

qspec_H1 = None
qspec_L1 = None

try:
    print("Computing Q-transform for H1 strain data...")
    qspec_H1 = strain_H1.q_transform(qrange=qrange, frange=frange)
    print("Q-transform for H1 computed successfully.")
    # Save to disk (must specify path for Spectrogram)
    qspec_H1.write(qspec_H1_path, path='/spectrogram', overwrite=True)
    print(f"H1 Q-transform saved to {qspec_H1_path}")
except Exception as e:
    print(f"Error computing or saving Q-transform for H1: {e}")

try:
    print("Computing Q-transform for L1 strain data...")
    qspec_L1 = strain_L1.q_transform(qrange=qrange, frange=frange)
    print("Q-transform for L1 computed successfully.")
    # Save to disk (must specify path for Spectrogram)
    qspec_L1.write(qspec_L1_path, path='/spectrogram', overwrite=True)
    print(f"L1 Q-transform saved to {qspec_L1_path}")
except Exception as e:
    print(f"Error computing or saving Q-transform for L1: {e}")

if qspec_H1 is None or qspec_L1 is None:
    print("Critical error: Could not compute both H1 and L1 Q-transforms. Exiting.")
    exit(1)

# =========================
# 3. Visualization
# =========================
print("\n" + "="*40)
print("Step 3: Plotting Q-transform spectrograms")
print("="*40)

def plot_q_transform(qspec, detector_label, output_dir):
    try:
        print(f"Plotting Q-transform spectrogram for {detector_label}...")
        fig = qspec.plot(norm='log', cmap='viridis')
        ax = fig.gca()
        # Set plot title and labels
        ax.set_title(f"{detector_label} Q-transform Spectrogram ({GW_EVENT})")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        # Adjust colorbar
        cbar = ax.images[-1].colorbar
        cbar.set_label('Normalized energy')
        plt.tight_layout()
        # Save figure
        fig_path = os.path.join(output_dir, f"{detector_label}_qtransform.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Q-transform spectrogram for {detector_label} saved to {fig_path}")
        plt.show()
        print(f"Q-transform spectrogram for {detector_label} displayed successfully.")
    except Exception as e:
        print(f"Error plotting Q-transform for {detector_label}: {e}")

plot_q_transform(qspec_H1, "H1", output_dir)
plot_q_transform(qspec_L1, "L1", output_dir)

print("\nAll steps completed successfully. Results are saved in the 'gw150914_outputs' directory.")