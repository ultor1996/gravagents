# =========================
# GW150914 Q-transform Analysis Script
# =========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# ---- Configuration ----
START_TIME = 1126259446
DURATION = 32
Q_RANGE = (4, 64)
F_RANGE = (20, 512)  # Hz

OUTPUT_DIR = "gw150914_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 1. Download Strain Data ----
print("=== Step 1: Downloading strain data for GW150914 ===")
strain_H1 = None
strain_L1 = None

try:
    print("Fetching H1 strain data...")
    strain_H1 = TimeSeries.fetch_open_data('H1', START_TIME, START_TIME + DURATION)
    print("Successfully fetched H1 strain data.")
    # Save strain data for reproducibility
    strain_H1.write(os.path.join(OUTPUT_DIR, "strain_H1.gwf"), format='gwf')
except Exception as e:
    print(f"Error fetching H1 data: {e}")

try:
    print("Fetching L1 strain data...")
    strain_L1 = TimeSeries.fetch_open_data('L1', START_TIME, START_TIME + DURATION)
    print("Successfully fetched L1 strain data.")
    # Save strain data for reproducibility
    strain_L1.write(os.path.join(OUTPUT_DIR, "strain_L1.gwf"), format='gwf')
except Exception as e:
    print(f"Error fetching L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    print("ERROR: Failed to fetch both H1 and L1 strain data. Exiting.")
    exit(1)

# ---- 2. Compute Q-transform ----
print("\n=== Step 2: Computing Q-transform ===")
q_H1 = None
q_L1 = None

try:
    print("Computing Q-transform for H1...")
    q_H1 = strain_H1.q_transform(qrange=Q_RANGE, frange=F_RANGE)
    print("Q-transform for H1 computed successfully.")
    # Save Q-transform object (using pickle, as GWpy objects are not natively serializable)
    with open(os.path.join(OUTPUT_DIR, "q_H1.pkl"), "wb") as f:
        pickle.dump(q_H1, f)
except Exception as e:
    print(f"Error computing Q-transform for H1: {e}")

try:
    print("Computing Q-transform for L1...")
    q_L1 = strain_L1.q_transform(qrange=Q_RANGE, frange=F_RANGE)
    print("Q-transform for L1 computed successfully.")
    with open(os.path.join(OUTPUT_DIR, "q_L1.pkl"), "wb") as f:
        pickle.dump(q_L1, f)
except Exception as e:
    print(f"Error computing Q-transform for L1: {e}")

if q_H1 is None or q_L1 is None:
    print("ERROR: Failed to compute Q-transform for both H1 and L1. Exiting.")
    exit(1)

# ---- 3. Plot Q-transform Spectrograms ----
print("\n=== Step 3: Plotting Q-transform spectrograms ===")

def plot_q_transform(q, detector_label, output_dir):
    try:
        print(f"Plotting Q-transform spectrogram for {detector_label}...")
        fig = q.plot()
        ax = fig.gca()
        ax.set_title(f"{detector_label} Q-transform Spectrogram (GW150914)")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        # Add colorbar with normalized energy label
        cbar = fig.colorbar(label="Normalized energy")
        # Save figure
        fig_path = os.path.join(output_dir, f"{detector_label}_q_transform.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"{detector_label} Q-transform spectrogram saved to {fig_path}")
        plt.show()
        print(f"{detector_label} Q-transform spectrogram plotted successfully.")
    except Exception as e:
        print(f"Error plotting {detector_label} Q-transform spectrogram: {e}")

plot_q_transform(q_H1, "H1", OUTPUT_DIR)
plot_q_transform(q_L1, "L1", OUTPUT_DIR)

print("\n=== Analysis complete. Results saved in:", OUTPUT_DIR, "===")