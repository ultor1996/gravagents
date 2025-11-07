# GW170817 Time-Frequency Analysis Integrated Script

# =========================
# Imports and Configuration
# =========================
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Parameters
# =========================
center_gps = 1187008882.43
duration = 1.0  # seconds for data retrieval
half_duration = duration / 2
start_time = center_gps - half_duration
end_time = center_gps + half_duration

# For visualization and Q-transform
window = 2.0  # seconds for plotting and Q-transform
half_window = window / 2
plot_start = center_gps - half_window
plot_end = center_gps + half_window

# Preprocessing filter parameters
highpass_freq = 30.0  # Hz
lowpass_freq = 250.0  # Hz

# Q-transform parameters
qrange = (8, 64)
frange = (30, 500)  # Hz

# Detector names
detectors = {'H1': 'H1', 'L1': 'L1'}

# =========================
# Task 1: Data Loading
# =========================
print("="*60)
print("TASK 1: Downloading 1-second raw strain data for GW170817...")
strain_data = {}

for det, det_name in detectors.items():
    print(f"Attempting to download {duration}-second strain data for {det_name} from {start_time} to {end_time}...")
    try:
        # FIX: Remove deprecated 'cache' argument as per GWpy API change
        ts = TimeSeries.get(f'{det_name}:LOSC-STRAIN', start_time, end_time)
        strain_data[det_name] = ts
        print(f"Successfully downloaded strain data for {det_name}.")
    except Exception as e:
        print(f"Error downloading strain data for {det_name}: {e}")
        strain_data[det_name] = None

if all(v is None for v in strain_data.values()):
    print("ERROR: No strain data could be downloaded for any detector. Exiting.")
    exit(1)

# =========================
# Task 2: Preprocessing
# =========================
print("\n" + "="*60)
print("TASK 2: Preprocessing (whitening and bandpass filtering)...")
preprocessed_strain = {}

for det in ['H1', 'L1']:
    print(f"\nStarting preprocessing for {det}...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"No strain data found for {det}; skipping preprocessing.")
        preprocessed_strain[det] = None
        continue
    try:
        print(f"Whitening {det} data...")
        ts_white = ts.whiten()
        print(f"Applying high-pass filter at {highpass_freq} Hz to {det} data...")
        ts_hp = ts_white.highpass(highpass_freq)
        print(f"Applying low-pass filter at {lowpass_freq} Hz to {det} data...")
        ts_bp = ts_hp.lowpass(lowpass_freq)
        preprocessed_strain[det] = ts_bp
        print(f"Preprocessing complete for {det}.")
    except Exception as e:
        print(f"Error during preprocessing for {det}: {e}")
        preprocessed_strain[det] = None

if all(v is None for v in preprocessed_strain.values()):
    print("ERROR: Preprocessing failed for all detectors. Exiting.")
    exit(1)

# =========================
# Task 3: Time-Domain Visualization
# =========================
print("\n" + "="*60)
print("TASK 3: Time-domain visualization of processed strain data...")

colors = {'H1': 'C0', 'L1': 'C1'}
labels = {'H1': 'Hanford (H1)', 'L1': 'Livingston (L1)'}
plotted_any = False

plt.figure(figsize=(10, 6))

for det in ['H1', 'L1']:
    ts = preprocessed_strain.get(det)
    if ts is None:
        print(f"Warning: No preprocessed data for {det}; skipping plot.")
        continue
    try:
        # Crop to 2-second window
        ts_window = ts.crop(plot_start, plot_end)
        # Time axis relative to merger
        t = ts_window.times.value - center_gps
        plt.plot(t, ts_window.value, label=labels[det], color=colors[det])
        plotted_any = True
        print(f"Plotted {det} data.")
    except Exception as e:
        print(f"Error plotting {det} data: {e}")

if not plotted_any:
    print("No data available to plot. Skipping visualization.")
else:
    plt.axvline(0, color='k', linestyle='--', label='Merger Time (t=0)')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened & Bandpassed Strain')
    plt.title('GW170817: Whitened and Bandpassed Strain vs. Time\n(H1 and L1, 2-second window)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# Task 4: Q-Transform Spectrograms
# =========================
print("\n" + "="*60)
print("TASK 4: Q-transform spectrograms for each detector...")

for det in ['H1', 'L1']:
    ts = preprocessed_strain.get(det)
    if ts is None:
        print(f"Warning: No preprocessed data for {det}; skipping Q-transform.")
        continue
    try:
        print(f"\nCropping {det} data to {window}-second window around merger...")
        ts_window = ts.crop(center_gps - half_window, center_gps + half_window)
        print(f"Computing Q-transform for {det} (Q={qrange}, f={frange} Hz)...")
        qspec = ts_window.q_transform(outseg=(center_gps - half_window, center_gps + half_window),
                                      qrange=qrange, frange=frange)
        print(f"Plotting Q-transform spectrogram for {det}...")
        fig = qspec.plot(figsize=(10, 6), vmin=0.0001, vmax=0.01, cmap='viridis')
        ax = fig.gca()
        # Mark merger time (t=0)
        ax.axvline(center_gps, color='w', linestyle='--', label='Merger Time')
        ax.set_title(f'{det} Q-transform Spectrogram\nGW170817 (2s window, Q={qrange}, {frange[0]}–{frange[1]} Hz)')
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating Q-transform for {det}: {e}")

print("\nAnalysis complete.")