# --- Imports ---
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parameters and Setup ---
event_gps = 1180922494.5  # GW170608 GPS time
duration = 64  # seconds (±32s)
half_window = duration // 2
start_time = event_gps - half_window
end_time = event_gps + half_window
detectors = ['H1', 'L1']

# Filtering parameters
low_freq = 35
high_freq = 350

# Plotting and q-transform parameters
zoom_window = 0.2  # seconds for strain plot
q_window = 0.5     # seconds for q-transform
plot_start = event_gps - zoom_window
plot_end = event_gps + zoom_window
q_start = event_gps - q_window
q_end = event_gps + q_window
q_kwargs = {
    'outseg': (q_start, q_end),
    'logf': True,
    'qrange': (8, 64),
    'frange': (20, 400),
    'stride': 0.01
}

# Output directory
output_dir = "gw170608_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download Strain Data ---
print("\n=== Task 1: Downloading strain data from GWOSC ===")
strain_data = {}
for det in detectors:
    try:
        print(f"Fetching {duration}s of strain data for {det} from GWOSC...")
        strain = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        strain_data[det] = strain
        # Save raw data for reproducibility
        strain.write(os.path.join(output_dir, f"{det}_strain_raw.txt"), format='txt')
        print(f"Successfully fetched and saved data for {det}.")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

strain_H1 = strain_data['H1']
strain_L1 = strain_data['L1']

# --- Task 2: Apply Bandpass Filter ---
print("\n=== Task 2: Applying bandpass filter ({}-{} Hz) ===".format(low_freq, high_freq))
filtered_strain = {}
for det, strain in zip(detectors, [strain_H1, strain_L1]):
    try:
        if strain is None:
            print(f"Warning: No strain data available for {det}, skipping filtering.")
            filtered_strain[det] = None
            continue
        print(f"Applying bandpass filter to {det} data...")
        filtered = strain.bandpass(low_freq, high_freq)
        filtered_strain[det] = filtered
        # Save filtered data
        filtered.write(os.path.join(output_dir, f"{det}_strain_filtered.txt"), format='txt')
        print(f"Filtering complete and data saved for {det}.")
    except Exception as e:
        print(f"Error filtering data for {det}: {e}")
        filtered_strain[det] = None

filtered_strain_H1 = filtered_strain['H1']
filtered_strain_L1 = filtered_strain['L1']

# --- Task 3: Plot Filtered Strain vs Time ---
print("\n=== Task 3: Plotting filtered strain vs time (±{:.1f}s) ===".format(zoom_window))
plt.figure(figsize=(10, 6))
plot_data = {'H1': filtered_strain_H1, 'L1': filtered_strain_L1}
plotted = False
for det, strain in plot_data.items():
    try:
        if strain is None:
            print(f"Warning: No filtered strain data for {det}, skipping plot.")
            continue
        print(f"Plotting filtered strain for {det}...")
        cropped = strain.crop(plot_start, plot_end)
        plt.plot(
            cropped.times.value - event_gps,
            cropped.value,
            label=f"{det} (filtered)"
        )
        plotted = True
    except Exception as e:
        print(f"Error plotting data for {det}: {e}")

plt.xlabel("Time (s) relative to GW170608")
plt.ylabel("Strain")
plt.title("Filtered Strain around GW170608 (±{:.1f}s)".format(zoom_window))
plt.legend()
plt.grid(True)
plt.tight_layout()
strain_plot_path = os.path.join(output_dir, "filtered_strain_zoom.png")
if plotted:
    plt.savefig(strain_plot_path)
    print(f"Filtered strain plot saved to {strain_plot_path}")
    plt.show()
else:
    print("No strain data available for plotting.")

# --- Task 4: Q-transform Spectrograms ---
print("\n=== Task 4: Generating q-transform spectrograms (±{:.1f}s) ===".format(q_window))
for det, strain in plot_data.items():
    try:
        if strain is None:
            print(f"Warning: No filtered strain data for {det}, skipping q-transform.")
            continue
        print(f"Generating q-transform spectrogram for {det}...")
        q = strain.q_transform(**q_kwargs)
        fig = q.plot()
        ax = fig.gca()
        ax.set_title(f"{det} Q-transform around GW170608")
        ax.set_xlim(q_start, q_end)
        ax.set_ylim(20, 400)
        ax.axvline(event_gps, color='r', linestyle='--', label='GW170608')
        ax.legend()
        plt.tight_layout()
        q_plot_path = os.path.join(output_dir, f"{det}_qtransform.png")
        plt.savefig(q_plot_path)
        print(f"Q-transform plot saved to {q_plot_path}")
        plt.show()
        print(f"Q-transform plot complete for {det}.")
    except Exception as e:
        print(f"Error generating q-transform for {det}: {e}")

print("\n=== Analysis complete. Outputs saved in '{}' ===".format(output_dir))