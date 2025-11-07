# ==========================
# GW170817 Time-Frequency Analysis Pipeline
# ==========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import sys

# ---- Constants ----
GW170817_GPS = 1187008882.43
RAW_DURATION = 1  # seconds
PLOT_WINDOW_SEC = 1.0  # For 2s window centered on merger
QTRANSFORM_QRANGE = (8, 32)
QTRANSFORM_FRANGE = (30, 250)
H1_LABEL = 'H1 (Hanford)'
L1_LABEL = 'L1 (Livingston)'

# ---- Output Paths ----
RAW_H1_FILE = "strain_H1_raw.hdf5"
RAW_L1_FILE = "strain_L1_raw.hdf5"
PROC_H1_FILE = "strain_H1_proc.hdf5"
PROC_L1_FILE = "strain_L1_proc.hdf5"
STRAIN_PLOT_FILE = "gw170817_strain_timeseries.png"
QTRANSFORM_H1_PLOT_FILE = "gw170817_qtransform_H1.png"
QTRANSFORM_L1_PLOT_FILE = "gw170817_qtransform_L1.png"

# ==========================
# 1. Data Loading
# ==========================
print("="*60)
print("Step 1: Downloading 1 second of raw strain data centered on GW170817...")
start = GW170817_GPS - RAW_DURATION / 2
end = GW170817_GPS + RAW_DURATION / 2

strain_H1 = None
strain_L1 = None

try:
    print(f"Downloading 1s of H1 strain data from {start} to {end} (centered on {GW170817_GPS})...")
    strain_H1 = TimeSeries.get('H1:LOSC-STRAIN', start, end, source='LOSC')
    print("H1 data download complete.")
    strain_H1.write(RAW_H1_FILE, format='hdf5')
    print(f"Raw H1 data saved to {RAW_H1_FILE}")
except Exception as e:
    print(f"ERROR: Failed to download Hanford (H1) data: {e}")
    sys.exit(1)

try:
    print(f"Downloading 1s of L1 strain data from {start} to {end} (centered on {GW170817_GPS})...")
    strain_L1 = TimeSeries.get('L1:LOSC-STRAIN', start, end, source='LOSC')
    print("L1 data download complete.")
    strain_L1.write(RAW_L1_FILE, format='hdf5')
    print(f"Raw L1 data saved to {RAW_L1_FILE}")
except Exception as e:
    print(f"ERROR: Failed to download Livingston (L1) data: {e}")
    sys.exit(1)

# ==========================
# 2. Preprocessing (Whitening + Bandpass)
# ==========================
print("="*60)
print("Step 2: Preprocessing strain data (whitening and bandpass filtering)...")

strain_H1_proc = None
strain_L1_proc = None

try:
    print("Whitening H1 strain data...")
    strain_H1_white = strain_H1.whiten()
    print("Applying high-pass filter (30 Hz) to H1...")
    strain_H1_hp = strain_H1_white.highpass(30)
    print("Applying low-pass filter (250 Hz) to H1...")
    strain_H1_proc = strain_H1_hp.lowpass(250)
    print("H1 preprocessing complete.")
    strain_H1_proc.write(PROC_H1_FILE, format='hdf5')
    print(f"Processed H1 data saved to {PROC_H1_FILE}")
except Exception as e:
    print(f"ERROR: Failed to preprocess Hanford (H1) data: {e}")
    sys.exit(1)

try:
    print("Whitening L1 strain data...")
    strain_L1_white = strain_L1.whiten()
    print("Applying high-pass filter (30 Hz) to L1...")
    strain_L1_hp = strain_L1_white.highpass(30)
    print("Applying low-pass filter (250 Hz) to L1...")
    strain_L1_proc = strain_L1_hp.lowpass(250)
    print("L1 preprocessing complete.")
    strain_L1_proc.write(PROC_L1_FILE, format='hdf5')
    print(f"Processed L1 data saved to {PROC_L1_FILE}")
except Exception as e:
    print(f"ERROR: Failed to preprocess Livingston (L1) data: {e}")
    sys.exit(1)

# ==========================
# 3. Strain Time Series Visualization
# ==========================
print("="*60)
print("Step 3: Visualizing processed strain vs. time for both detectors...")

plot_start = GW170817_GPS - PLOT_WINDOW_SEC
plot_end = GW170817_GPS + PLOT_WINDOW_SEC

try:
    print("Cropping H1 data for plotting...")
    strain_H1_plot = strain_H1_proc.crop(plot_start, plot_end)
    print("Cropping L1 data for plotting...")
    strain_L1_plot = strain_L1_proc.crop(plot_start, plot_end)
except Exception as e:
    print(f"ERROR: Failed to crop data for plotting: {e}")
    sys.exit(1)

# Generate time arrays relative to merger
time_H1 = strain_H1_plot.times.value - GW170817_GPS
time_L1 = strain_L1_plot.times.value - GW170817_GPS

try:
    plt.figure(figsize=(12, 6))
    plt.plot(time_H1, strain_H1_plot.value, label=H1_LABEL, color='C0')
    plt.plot(time_L1, strain_L1_plot.value, label=L1_LABEL, color='C1')
    plt.axvline(0, color='k', linestyle='--', label='Merger Time')
    plt.title('Processed Strain Data around GW170817 Merger')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened & Filtered Strain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(STRAIN_PLOT_FILE, dpi=150)
    plt.show()
    print(f"Strain vs. time visualization complete. Figure saved to {STRAIN_PLOT_FILE}")
except Exception as e:
    print(f"ERROR: Failed to plot strain time series: {e}")
    sys.exit(1)

# ==========================
# 4. Q-transform Spectrograms
# ==========================
print("="*60)
print("Step 4: Generating Q-transform spectrograms for both detectors...")

try:
    print("Computing Q-transform for H1...")
    qspec_H1 = strain_H1_proc.q_transform(frange=QTRANSFORM_FRANGE, qrange=QTRANSFORM_QRANGE)
    print("Q-transform for H1 complete.")
except Exception as e:
    print(f"ERROR: Failed to compute Q-transform for H1: {e}")
    sys.exit(1)

try:
    print("Computing Q-transform for L1...")
    qspec_L1 = strain_L1_proc.q_transform(frange=QTRANSFORM_FRANGE, qrange=QTRANSFORM_QRANGE)
    print("Q-transform for L1 complete.")
except Exception as e:
    print(f"ERROR: Failed to compute Q-transform for L1: {e}")
    sys.exit(1)

# Plot H1 Q-transform
try:
    print("Plotting Q-transform spectrogram for H1...")
    fig1 = qspec_H1.plot(figsize=(12, 5), cmap='viridis')
    ax1 = fig1.gca()
    ax1.set_title('H1 Q-transform Spectrogram (GW170817)')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [s] relative to merger')
    # Adjust x-axis to show time relative to merger
    xlim = ax1.get_xlim()
    ax1.set_xlim(xlim[0] - GW170817_GPS, xlim[1] - GW170817_GPS)
    plt.tight_layout()
    plt.savefig(QTRANSFORM_H1_PLOT_FILE, dpi=150)
    plt.show()
    print(f"H1 Q-transform spectrogram plotted. Figure saved to {QTRANSFORM_H1_PLOT_FILE}")
except Exception as e:
    print(f"ERROR: Failed to plot H1 Q-transform: {e}")
    sys.exit(1)

# Plot L1 Q-transform
try:
    print("Plotting Q-transform spectrogram for L1...")
    fig2 = qspec_L1.plot(figsize=(12, 5), cmap='viridis')
    ax2 = fig2.gca()
    ax2.set_title('L1 Q-transform Spectrogram (GW170817)')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [s] relative to merger')
    xlim = ax2.get_xlim()
    ax2.set_xlim(xlim[0] - GW170817_GPS, xlim[1] - GW170817_GPS)
    plt.tight_layout()
    plt.savefig(QTRANSFORM_L1_PLOT_FILE, dpi=150)
    plt.show()
    print(f"L1 Q-transform spectrogram plotted. Figure saved to {QTRANSFORM_L1_PLOT_FILE}")
except Exception as e:
    print(f"ERROR: Failed to plot L1 Q-transform: {e}")
    sys.exit(1)

print("="*60)
print("All steps completed successfully.")