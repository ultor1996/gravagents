# =========================
# GW170817 Time-Frequency Analysis Integrated Script
# =========================

# ---- Imports ----
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# ---- Constants ----
MERGER_GPS = 1187008882.43
RAW_WINDOW = 0.5   # seconds before and after for 1s window (raw data)
PLOT_WINDOW = 1.0  # seconds before and after for 2s window (plots)
CHANNELS = {
    'H1': 'H1:GWOSC-16KHZ_R1_STRAIN',
    'L1': 'L1:GWOSC-16KHZ_R1_STRAIN'
}
Q_RANGE = (8, 64)
F_RANGE = (30, 250)

# ---- Output File Names ----
RAW_DATA_FILE = "gw170817_raw_strain.hdf5"
PREPROCESSED_DATA_FILE = "gw170817_preprocessed_strain.hdf5"
TIME_DOMAIN_PLOT_FILE = "gw170817_strain_time_domain.png"
QTRANSFORM_PLOT_TEMPLATE = "gw170817_qtransform_{det}.png"

# ---- Utility Functions ----
def save_timeseries_dict_hdf5(ts_dict, filename):
    """Save a dictionary of GWpy TimeSeries to HDF5."""
    import h5py
    with h5py.File(filename, 'w') as f:
        for det, ts in ts_dict.items():
            if ts is not None:
                grp = f.create_group(det)
                grp.create_dataset('times', data=ts.times.value)
                grp.create_dataset('strain', data=ts.value)
                grp.attrs['sample_rate'] = ts.sample_rate.value
                grp.attrs['t0'] = ts.t0.value
    print(f"Saved data to {filename}")

def load_timeseries_dict_hdf5(filename):
    """Load a dictionary of GWpy TimeSeries from HDF5."""
    import h5py
    ts_dict = {}
    with h5py.File(filename, 'r') as f:
        for det in f.keys():
            grp = f[det]
            times = grp['times'][:]
            strain = grp['strain'][:]
            sample_rate = grp.attrs['sample_rate']
            t0 = grp.attrs['t0']
            ts = TimeSeries(strain, times=times, sample_rate=sample_rate, t0=t0)
            ts_dict[det] = ts
    print(f"Loaded data from {filename}")
    return ts_dict

# =========================
# 1. Data Loading
# =========================
print("\n=== 1. DATA LOADING ===")
start_time = MERGER_GPS - RAW_WINDOW
end_time = MERGER_GPS + RAW_WINDOW
strain_data = {}

for det, channel in CHANNELS.items():
    print(f"Attempting to download {det} strain data from {start_time} to {end_time}...")
    try:
        ts = TimeSeries.get(channel, start_time, end_time, cache=True)
        strain_data[det] = ts
        print(f"Successfully downloaded {det} data: {len(ts)} samples, sample rate {ts.sample_rate.value} Hz")
    except Exception as e:
        print(f"Error downloading {det} data: {e}")
        strain_data[det] = None

# Save raw data
try:
    save_timeseries_dict_hdf5(strain_data, RAW_DATA_FILE)
except Exception as e:
    print(f"Warning: Could not save raw data to HDF5: {e}")

# =========================
# 2. Preprocessing
# =========================
print("\n=== 2. PREPROCESSING ===")
preprocessed_data = {}

for det in ['H1', 'L1']:
    print(f"\nStarting preprocessing for {det}...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"No data found for {det}, skipping preprocessing.")
        preprocessed_data[det] = None
        continue
    try:
        print(f"Whitening {det} data...")
        ts_white = ts.whiten()
        print(f"Applying highpass filter at 30 Hz to {det} data...")
        ts_hp = ts_white.highpass(30)
        print(f"Applying lowpass filter at 250 Hz to {det} data...")
        ts_bp = ts_hp.lowpass(250)
        preprocessed_data[det] = ts_bp
        print(f"Preprocessing complete for {det}.")
    except Exception as e:
        print(f"Error during preprocessing for {det}: {e}")
        preprocessed_data[det] = None

# Save preprocessed data
try:
    save_timeseries_dict_hdf5(preprocessed_data, PREPROCESSED_DATA_FILE)
except Exception as e:
    print(f"Warning: Could not save preprocessed data to HDF5: {e}")

# =========================
# 3. Time-Domain Visualization
# =========================
print("\n=== 3. TIME-DOMAIN VISUALIZATION ===")
plt.figure(figsize=(10, 6))
colors = {'H1': 'C0', 'L1': 'C1'}
labels = {'H1': 'Hanford (H1)', 'L1': 'Livingston (L1)'}
data_found = False

for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"Warning: No preprocessed data for {det}, skipping plot.")
        continue
    try:
        # Extract 2-second window centered on merger
        ts_window = ts.crop(MERGER_GPS - PLOT_WINDOW, MERGER_GPS + PLOT_WINDOW)
        # Time axis relative to merger
        time_rel = ts_window.times.value - MERGER_GPS
        plt.plot(time_rel, ts_window.value, label=labels[det], color=colors[det])
        data_found = True
        print(f"Plotted {det} data.")
    except Exception as e:
        print(f"Error plotting {det} data: {e}")

if not data_found:
    print("No data available to plot. Skipping time-domain visualization.")
else:
    plt.axvline(0, color='k', linestyle='--', label='Merger Time')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened, Bandpassed Strain')
    plt.title('GW170817: Whitened & Bandpassed Strain vs. Time\n(H1 and L1, 2-second window)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(TIME_DOMAIN_PLOT_FILE, dpi=150)
    print(f"Time-domain plot saved as {TIME_DOMAIN_PLOT_FILE}")
    plt.show()

# =========================
# 4. Q-Transform Spectrograms
# =========================
print("\n=== 4. Q-TRANSFORM SPECTROGRAMS ===")
for det in ['H1', 'L1']:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"Warning: No preprocessed data for {det}, skipping Q-transform.")
        continue
    try:
        print(f"\nGenerating Q-transform for {det}...")
        # Crop to 2-second window
        ts_win = ts.crop(MERGER_GPS - PLOT_WINDOW, MERGER_GPS + PLOT_WINDOW)
        # Compute Q-transform
        qspec = ts_win.q_transform(outseg=(MERGER_GPS - PLOT_WINDOW, MERGER_GPS + PLOT_WINDOW),
                                   qrange=Q_RANGE, frange=F_RANGE)
        print(f"Q-transform computed for {det}. Plotting spectrogram...")
        # Plot
        fig = qspec.plot(figsize=(10, 6), vmin=0, vmax=5)
        ax = fig.gca()
        ax.set_title(f'{det} Q-transform Spectrogram (GW170817)\n2s window, {F_RANGE[0]}–{F_RANGE[1]} Hz, Q={Q_RANGE[0]}–{Q_RANGE[1]}')
        ax.axvline(MERGER_GPS, color='w', linestyle='--', label='Merger Time')
        ax.set_xlabel('GPS Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend()
        fig.colorbar(ax.images[0], ax=ax, label='Normalized energy')
        plt.tight_layout()
        plot_file = QTRANSFORM_PLOT_TEMPLATE.format(det=det)
        plt.savefig(plot_file, dpi=150)
        print(f"Spectrogram for {det} saved as {plot_file}")
        plt.show()
    except Exception as e:
        print(f"Error generating or plotting Q-transform for {det}: {e}")

print("\n=== GW170817 Time-Frequency Analysis Complete ===")