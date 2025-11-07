# ============================================================
# GW150914 Q-Transform Spectroscopy with Normalized Energy Bar
# ============================================================

import os
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# -------------------------------
# Section 1: Data Download
# -------------------------------

print("========== GW150914 Q-Transform Spectroscopy ==========")
print("Step 1: Downloading strain data for H1 and L1 detectors...")

# Define GPS interval for GW150914
gps_start = 1126259446
gps_end = 1126259462
detectors = ['H1', 'L1']

# Dictionary to store strain data
strain_data = {}

for det in detectors:
    print(f"  - Downloading strain data for {det} from GPS {gps_start} to {gps_end}...")
    try:
        strain = TimeSeries.fetch_open_data(det, gps_start, gps_end, cache=True)
        strain_data[det] = strain
        print(f"    -> Successfully downloaded strain data for {det}.")
    except Exception as e:
        print(f"    !! Failed to download strain data for {det}: {e}")
        strain_data[det] = None

# Optionally save strain data for reproducibility
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

# Q-transform parameters
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

print("\nStep 3: Plotting Q-transform spectrograms with normalized energy colorbar...")

# Event GPS time for GW150914
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
        # Plot Q-transform with log normalization for energy
        fig = qtrans.plot(norm='log')
        ax = fig.gca()
        # Add vertical line for event time
        ax.axvline(event_gps_time, color='red', linestyle='--', label='GW150914 Event')
        # Set axis labels and title
        ax.set_xlabel("Time (GPS seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{det} Q-transform Spectrogram\nGW150914")
        # Add legend for event annotation
        ax.legend()
        # Ensure colorbar is visible and labeled
        # GWpy's plot returns a Figure with an AxesImage in ax.images[0]
        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.set_label('Normalized energy')
        # Save the plot
        plot_filename = os.path.join(output_dir, f"GW150914_{det}_qtransform.png")
        fig.savefig(plot_filename, dpi=150)
        print(f"    -> Q-transform spectrogram for {det} saved to {plot_filename}.")
        # Show the plot
        plt.show()
        plt.close(fig)
        print(f"    -> Displayed Q-transform spectrogram for {det}.")
    except Exception as e:
        print(f"    !! Error plotting Q-transform for {det}: {e}")

print("\n========== Workflow complete. ==========")