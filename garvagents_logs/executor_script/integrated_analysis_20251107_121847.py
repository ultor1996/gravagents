# ============================================================
# GW150914 Q-Transform Spectroscopy: H1 and L1 Detectors
# ============================================================

import os
import sys
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# -------------------------------
# Section 1: Data Download
# -------------------------------

print("========== GW150914 Q-Transform Spectroscopy ==========")
print("Step 1: Downloading strain data for H1 and L1 detectors...")

# Define GPS start and end times for GW150914
gps_start = 1126259446
gps_end = 1126259462
detectors = ['H1', 'L1']

# Dictionary to store strain data for each detector
strain_data = {}

for det in detectors:
    print(f"  - Attempting to download strain data for detector {det} from GPS {gps_start} to {gps_end}...")
    try:
        strain = TimeSeries.fetch_open_data(det, gps_start, gps_end, cache=True)
        strain_data[det] = strain
        print(f"    -> Successfully downloaded strain data for {det}.")
    except Exception as e:
        print(f"    !! Error downloading strain data for {det}: {e}")
        strain_data[det] = None

# Save strain data to disk (optional, for reproducibility)
for det, strain in strain_data.items():
    if strain is not None:
        try:
            filename = f"GW150914_{det}_strain.gwf"
            strain.write(filename)
            print(f"    -> Strain data for {det} saved to {filename}.")
        except Exception as e:
            print(f"    !! Error saving strain data for {det}: {e}")

# -------------------------------
# Section 2: Q-Transform Computation
# -------------------------------

print("\nStep 2: Computing Q-transform for each detector...")

# Parameters for Q-transform
qrange = (4, 64)
frange = (20, 500)

# Dictionary to store Q-transform results
qtransforms = {}

for det in detectors:
    print(f"  - Computing Q-transform for {det}...")
    strain = strain_data.get(det)
    if strain is None:
        print(f"    !! No strain data available for {det}, skipping Q-transform.")
        qtransforms[det] = None
        continue
    try:
        qtrans = strain.q_transform(qrange=qrange, frange=frange)
        qtransforms[det] = qtrans
        print(f"    -> Q-transform for {det} computed successfully.")
    except Exception as e:
        print(f"    !! Error computing Q-transform for {det}: {e}")
        qtransforms[det] = None

# -------------------------------
# Section 3: Visualization
# -------------------------------

print("\nStep 3: Plotting Q-transform spectrograms...")

# Event time for annotation (GW150914)
event_gps_time = 1126259462.422

# Create output directory for plots
output_dir = "qtransform_plots"
os.makedirs(output_dir, exist_ok=True)

for det in detectors:
    print(f"  - Plotting Q-transform spectrogram for {det}...")
    qtrans = qtransforms.get(det)
    if qtrans is None:
        print(f"    !! No Q-transform data available for {det}, skipping plot.")
        continue
    try:
        # Create the plot
        fig = qtrans.plot(norm='log', vmin=1e-24, vmax=1e-21)
        ax = fig.gca()
        # Annotate event time
        ax.axvline(event_gps_time, color='red', linestyle='--', label='GW150914 Event')
        # Add detector label and title
        ax.set_title(f"{det} Q-transform Spectrogram\nGW150914")
        ax.set_xlabel("Time (GPS seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend()
        # Save the plot
        plot_filename = os.path.join(output_dir, f"GW150914_{det}_qtransform.png")
        fig.savefig(plot_filename, dpi=150)
        print(f"    -> Q-transform spectrogram for {det} saved to {plot_filename}.")
        # Show the plot interactively
        plt.show()
        plt.close(fig)
        print(f"    -> Displayed Q-transform spectrogram for {det}.")
    except Exception as e:
        print(f"    !! Error plotting Q-transform for {det}: {e}")

print("\n========== Workflow complete. ==========")