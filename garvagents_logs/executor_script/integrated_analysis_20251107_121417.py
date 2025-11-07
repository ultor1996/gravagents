# =========================
# GW150914 Q-Transform Analysis Script
# =========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
import matplotlib.pyplot as plt
import sys
import os

# ---- Parameters ----
EVENT_GPS = 1126259462  # GW150914 GPS time
WINDOW = 4096           # seconds
START_TIME = EVENT_GPS - WINDOW // 2
END_TIME = EVENT_GPS + WINDOW // 2

QTRANSFORM_PARAMS = {
    'logfbins': 100,        # Number of frequency bins (logarithmic)
    'qrange': (4, 64),      # Range of Q values
    'frange': (20, 512),    # Frequency range in Hz
    'stride': 0.1,          # Time stride in seconds
}

OUTPUT_DIR = "gw150914_qtransform_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. Download Strain Data
# =========================
print("="*40)
print("Step 1: Downloading strain data for GW150914")
print("="*40)
strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {START_TIME} to {END_TIME} (GPS)...")
    strain_H1 = TimeSeries.fetch_open_data('H1', START_TIME, END_TIME, cache=True)
    print("H1 strain data downloaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to fetch H1 data: {e}")
    sys.exit(1)

try:
    print(f"Fetching L1 strain data from {START_TIME} to {END_TIME} (GPS)...")
    strain_L1 = TimeSeries.fetch_open_data('L1', START_TIME, END_TIME, cache=True)
    print("L1 strain data downloaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to fetch L1 data: {e}")
    sys.exit(1)

# =========================
# 2. Compute Q-Transform
# =========================
print("\n" + "="*40)
print("Step 2: Computing Q-transform for both detectors")
print("="*40)
qspec_H1 = None
qspec_L1 = None

try:
    print("Computing Q-transform for H1...")
    qspec_H1 = strain_H1.q_transform(**QTRANSFORM_PARAMS)
    print("Q-transform for H1 completed.")
except Exception as e:
    print(f"ERROR: Failed to compute Q-transform for H1: {e}")
    sys.exit(1)

try:
    print("Computing Q-transform for L1...")
    qspec_L1 = strain_L1.q_transform(**QTRANSFORM_PARAMS)
    print("Q-transform for L1 completed.")
except Exception as e:
    print(f"ERROR: Failed to compute Q-transform for L1: {e}")
    sys.exit(1)

# Optionally, save Q-transform data for reproducibility
try:
    qspec_H1.write(os.path.join(OUTPUT_DIR, "qspec_H1.gwf"), overwrite=True)
    qspec_L1.write(os.path.join(OUTPUT_DIR, "qspec_L1.gwf"), overwrite=True)
    print("Q-transform data saved to disk.")
except Exception as e:
    print(f"WARNING: Could not save Q-transform data: {e}")

# =========================
# 3. Plot Q-Transform Spectrograms
# =========================
print("\n" + "="*40)
print("Step 3: Plotting Q-transform spectrograms")
print("="*40)

def plot_q_transform(qspec, detector, event_time, output_dir):
    try:
        print(f"Plotting Q-transform for {detector}...")
        fig = qspec.plot(figsize=(10, 6), vmin=0, vmax=20)
        ax = fig.gca()
        ax.set_title(f"{detector} Q-transform around GW150914")
        ax.set_xlabel("Time (GPS)")
        ax.set_ylabel("Frequency [Hz]")
        # Mark the event time
        ax.axvline(event_time, color='red', linestyle='--', label='GW150914')
        ax.legend()
        plt.tight_layout()
        # Save figure
        fig_path = os.path.join(output_dir, f"qtransform_{detector}.png")
        fig.savefig(fig_path)
        print(f"Q-transform plot for {detector} saved to {fig_path}")
        plt.show()
        print(f"Q-transform plot for {detector} displayed.")
        return fig
    except Exception as e:
        print(f"ERROR: Failed to plot Q-transform for {detector}: {e}")
        return None

fig_H1 = plot_q_transform(qspec_H1, 'H1', EVENT_GPS, OUTPUT_DIR)
fig_L1 = plot_q_transform(qspec_L1, 'L1', EVENT_GPS, OUTPUT_DIR)

print("\nAll steps completed successfully.")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")