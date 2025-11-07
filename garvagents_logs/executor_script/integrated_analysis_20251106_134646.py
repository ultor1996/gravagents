# =========================
# GW150914 Q-transform Spectrograms for H1 and L1
# =========================

# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import os

# --- Configuration ---
gw150914_gps = 1126259462
window = 4  # seconds before and after the event
start_time = gw150914_gps - window
end_time = gw150914_gps + window

output_dir = "gw150914_qtrans_results"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download Strain Data ---
print("="*60)
print(f"Fetching GW150914 strain data from {start_time} to {end_time} for both H1 and L1 detectors...")

strain_H1 = None
strain_L1 = None

try:
    print("Downloading Hanford (H1) strain data...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("Hanford (H1) strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading Hanford (H1) data: {e}")

try:
    print("Downloading Livingston (L1) strain data...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("Livingston (L1) strain data downloaded successfully.")
except Exception as e:
    print(f"Error downloading Livingston (L1) data: {e}")

# --- Task 2: Q-transform for H1 ---
print("="*60)
print("Starting Q-transform computation and plotting for Hanford (H1) strain data...")

qspec_H1 = None
try:
    if strain_H1 is not None:
        print("Computing Q-transform for H1...")
        qspec_H1 = strain_H1.q_transform()
        print("Q-transform computed. Plotting spectrogram...")
        fig = qspec_H1.plot()
        ax = fig.gca()
        ax.set_title("Q-transform Spectrogram - Hanford (H1) GW150914")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, "qtransform_H1.png")
        fig.savefig(plot_filename)
        print(f"Q-transform spectrogram for H1 saved to {plot_filename}.")
        plt.show()
        print("Q-transform spectrogram for H1 displayed successfully.")
    else:
        print("strain_H1 data not available. Cannot compute Q-transform.")
except Exception as e:
    print(f"Error during Q-transform computation or plotting for H1: {e}")

# --- Task 3: Q-transform for L1 ---
print("="*60)
print("Starting Q-transform computation and plotting for Livingston (L1) strain data...")

qspec_L1 = None
try:
    if strain_L1 is not None:
        print("Computing Q-transform for L1...")
        qspec_L1 = strain_L1.q_transform()
        print("Q-transform computed. Plotting spectrogram...")
        fig = qspec_L1.plot()
        ax = fig.gca()
        ax.set_title("Q-transform Spectrogram - Livingston (L1) GW150914")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, "qtransform_L1.png")
        fig.savefig(plot_filename)
        print(f"Q-transform spectrogram for L1 saved to {plot_filename}.")
        plt.show()
        print("Q-transform spectrogram for L1 displayed successfully.")
    else:
        print("strain_L1 data not available. Cannot compute Q-transform.")
except Exception as e:
    print(f"Error during Q-transform computation or plotting for L1: {e}")

print("="*60)
print("GW150914 Q-transform analysis complete.")
print(f"All results saved in the '{output_dir}' directory.")