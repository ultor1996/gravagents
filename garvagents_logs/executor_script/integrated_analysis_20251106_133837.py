# =========================
# GW150914 Strain Analysis: Download, Whiten, and Plot
# =========================

# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
gw150914_gps = 1126259462  # GW150914 GPS time
duration = 8               # seconds of data to fetch
start_time = gw150914_gps - 4
end_time = gw150914_gps + 4

output_dir = "gw150914_results"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download Strain Data ---
print("="*60)
print(f"Fetching strain data for GW150914 from {start_time} to {end_time}...")

strain_H1 = None
strain_L1 = None

try:
    print("Downloading Hanford (H1) data...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("Hanford (H1) data downloaded successfully.")
    # Save raw data
    np.savetxt(os.path.join(output_dir, "strain_H1.txt"),
               np.column_stack((strain_H1.times.value, strain_H1.value)),
               header="Time(s) Strain")
    print("Hanford (H1) strain data saved to strain_H1.txt.")
except Exception as e:
    print(f"Error downloading Hanford (H1) data: {e}")

try:
    print("Downloading Livingston (L1) data...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("Livingston (L1) data downloaded successfully.")
    # Save raw data
    np.savetxt(os.path.join(output_dir, "strain_L1.txt"),
               np.column_stack((strain_L1.times.value, strain_L1.value)),
               header="Time(s) Strain")
    print("Livingston (L1) strain data saved to strain_L1.txt.")
except Exception as e:
    print(f"Error downloading Livingston (L1) data: {e}")

# --- Task 2: Whiten the Data ---
print("="*60)
print("Starting whitening of strain data...")

whitened_H1 = None
whitened_L1 = None

try:
    if strain_H1 is not None:
        print("Whitening Hanford (H1) strain data...")
        whitened_H1 = strain_H1.whiten()
        print("Hanford (H1) strain data whitened successfully.")
        # Save whitened data
        np.savetxt(os.path.join(output_dir, "whitened_H1.txt"),
                   np.column_stack((whitened_H1.times.value, whitened_H1.value)),
                   header="Time(s) Whitened_Strain")
        print("Whitened Hanford (H1) data saved to whitened_H1.txt.")
    else:
        print("Hanford (H1) strain data not available. Skipping whitening.")
except Exception as e:
    print(f"Error whitening Hanford (H1) data: {e}")

try:
    if strain_L1 is not None:
        print("Whitening Livingston (L1) strain data...")
        whitened_L1 = strain_L1.whiten()
        print("Livingston (L1) strain data whitened successfully.")
        # Save whitened data
        np.savetxt(os.path.join(output_dir, "whitened_L1.txt"),
                   np.column_stack((whitened_L1.times.value, whitened_L1.value)),
                   header="Time(s) Whitened_Strain")
        print("Whitened Livingston (L1) data saved to whitened_L1.txt.")
    else:
        print("Livingston (L1) strain data not available. Skipping whitening.")
except Exception as e:
    print(f"Error whitening Livingston (L1) data: {e}")

# --- Task 3: Plot Whitened Strain ---
print("="*60)
print("Starting plot of whitened strain data...")

def plot_whitened_strain(whitened_strain, detector_label, output_dir):
    try:
        print(f"Plotting whitened strain for {detector_label}...")
        fig = whitened_strain.plot()
        ax = fig.gca()
        ax.set_title(f"Whitened Strain Data - {detector_label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Whitened Strain")
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"whitened_{detector_label.replace(' ', '_')}.png")
        fig.savefig(plot_filename)
        print(f"Plot for {detector_label} saved to {plot_filename}.")
        plt.show()
        print(f"Plot for {detector_label} displayed successfully.")
    except Exception as e:
        print(f"Error plotting whitened strain for {detector_label}: {e}")

if whitened_H1 is not None:
    plot_whitened_strain(whitened_H1, "Hanford (H1)", output_dir)
else:
    print("Whitened Hanford (H1) data not available. Skipping plot.")

if whitened_L1 is not None:
    plot_whitened_strain(whitened_L1, "Livingston (L1)", output_dir)
else:
    print("Whitened Livingston (L1) data not available. Skipping plot.")

print("="*60)
print("GW150914 strain data analysis complete.")
print(f"All results saved in the '{output_dir}' directory.")