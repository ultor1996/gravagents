# ================================================================
# GW150914 Q-Transform Spectroscopy: H1 and L1 Detectors
# ================================================================

# ---------------------------
# Imports
# ---------------------------
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------
# Parameters and Setup
# ---------------------------
event_gps = 1126259462.4
segment_duration = 4  # seconds
half_segment = segment_duration / 2
start_time = event_gps - half_segment
end_time = event_gps + half_segment
detectors = ['H1', 'L1']

# Output directories
output_dir = "gw150914_outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 1. Download Strain Data
# ---------------------------
print("="*60)
print("Step 1: Downloading strain data for GW150914 (H1 and L1)")
print("="*60)
strain_data = {}

for det in detectors:
    try:
        print(f"Fetching strain data for {det} from {start_time} to {end_time}...")
        strain = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        strain_data[det] = strain
        # Save strain data to file for reproducibility
        strain.write(os.path.join(output_dir, f"{det}_strain.gwf"), format='gwf')
        print(f"Successfully fetched and saved data for {det}.")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

strain_H1 = strain_data.get('H1', None)
strain_L1 = strain_data.get('L1', None)

# ---------------------------
# 2. Compute Q-Transform
# ---------------------------
print("\n" + "="*60)
print("Step 2: Computing Q-transform spectrograms")
print("="*60)
frange = (20, 500)  # Hz
qrange = (4, 64)
qtransforms = {}

for det, strain in [('H1', strain_H1), ('L1', strain_L1)]:
    if strain is None:
        print(f"Strain data for {det} not available. Skipping Q-transform.")
        qtransforms[det] = None
        continue
    try:
        print(f"Computing Q-transform for {det}...")
        qspec = strain.q_transform(frange=frange, qrange=qrange)
        qtransforms[det] = qspec
        # Save Q-transform data as numpy array for reproducibility
        np.save(os.path.join(output_dir, f"{det}_qtransform.npy"), qspec.value)
        print(f"Q-transform for {det} computed and saved successfully.")
    except Exception as e:
        print(f"Error computing Q-transform for {det}: {e}")
        qtransforms[det] = None

qspec_H1 = qtransforms.get('H1', None)
qspec_L1 = qtransforms.get('L1', None)

# ---------------------------
# 3. Plot Q-Transform Spectrograms
# ---------------------------
print("\n" + "="*60)
print("Step 3: Plotting Q-transform spectrograms")
print("="*60)

def plot_q_transform(qspec, det, event_time, output_dir):
    if qspec is None:
        print(f"No Q-transform data available for {det}. Skipping plot.")
        return
    try:
        print(f"Plotting Q-transform for {det}...")
        fig = qspec.plot(figsize=(10, 6))
        ax = fig.gca()
        # Add vertical line at event time
        ax.axvline(event_time, color='red', linestyle='--', label='GW150914')
        # Set labels and title
        ax.set_title(f"{det} Q-transform Spectrogram around GW150914")
        ax.set_xlabel("GPS Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend()
        plt.tight_layout()
        # Save plot to file
        plot_path = os.path.join(output_dir, f"{det}_qtransform.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Displayed and saved Q-transform plot for {det} at {plot_path}.")
    except Exception as e:
        print(f"Error plotting Q-transform for {det}: {e}")

plot_q_transform(qspec_H1, 'H1', event_gps, output_dir)
plot_q_transform(qspec_L1, 'L1', event_gps, output_dir)

print("\nAll steps completed. Outputs saved in:", output_dir)