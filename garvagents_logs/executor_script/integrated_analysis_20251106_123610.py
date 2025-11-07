# ==========================
# GW170817 Time-Frequency Analysis Pipeline (FIXED)
# ==========================

# ---- Imports ----
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ---- Constants ----
GW170817_GPS = 1187008882.43
RAW_WINDOW_SEC = 0.5  # For 1s window centered on merger
PLOT_WINDOW_SEC = 1.0  # For 2s window centered on merger
QTRANSFORM_QRANGE = (5, 100)
QTRANSFORM_FRANGE = (30, 250)
H1_LABEL = 'H1 (Hanford)'
L1_LABEL = 'L1 (Livingston)'

# ---- Output Paths ----
RAW_H1_FILE = "strain_H1_raw.hdf5"
RAW_L1_FILE = "strain_L1_raw.hdf5"
PROC_H1_FILE = "strain_H1_proc.hdf5"
PROC_L1_FILE = "strain_L1_proc.hdf5"
STRAIN_PLOT_FILE = "gw170817_strain_timeseries.png"
QTRANSFORM_PLOT_FILE = "gw170817_qtransform_spectrograms.png"

# ==========================
# 1. Data Loading
# ==========================
print("="*60)
print("Step 1: Downloading 1 second of raw strain data centered on GW170817...")
start_time = GW170817_GPS - RAW_WINDOW_SEC
end_time = GW170817_GPS + RAW_WINDOW_SEC

strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching Hanford (H1) data: {start_time} to {end_time} (GPS)...")
    # Remove deprecated cache argument per database
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time)
    print("Hanford (H1) data downloaded successfully.")
    strain_H1.write(RAW_H1_FILE, format='hdf5')
    print(f"Raw H1 data saved to {RAW_H1_FILE}")
except Exception as e:
    print(f"ERROR: Failed to download Hanford (H1) data: {e}")
    sys.exit(1)

try:
    print(f"Fetching Livingston (L1) data: {start_time} to {end_time} (GPS)...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time)
    print("Livingston (L1) data downloaded successfully.")
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

def safe_whiten(strain, label):
    """
    Whiten the strain data, dynamically adjusting seg_len if needed.
    """
    duration = strain.duration.value
    # Default seg_len in GWpy is 4s, which is too long for 1s data
    # Use seg_len = min(0.5 * duration, duration) and ensure it's at least 0.25s
    seg_len = max(0.25, min(duration/2, duration))
    try:
        print(f"Whitening {label} data with seg_len={seg_len:.3f}s ...")
        # GWpy's whiten() accepts seg_len and seg_stride
        strain_white = strain.whiten(seg_len=seg_len, seg_stride=seg_len/2)
        return strain_white
    except Exception as e:
        print(f"ERROR: Whitening failed for {label}: {e}")
        sys.exit(1)

try:
    strain_H1_white = safe_whiten(strain_H1, H1_LABEL)
    print("Applying 30 Hz high-pass filter to H1...")
    strain_H1_hp = strain_H1_white.highpass(30)
    print("Applying 250 Hz low-pass filter to H1...")
    strain_H1_proc = strain_H1_hp.lowpass(250)
    print("Hanford (H1) preprocessing complete.")
    strain_H1_proc.write(PROC_H1_FILE, format='hdf5')
    print(f"Processed H1 data saved to {PROC_H1_FILE}")
except Exception as e:
    print(f"ERROR: Failed to preprocess Hanford (H1) data: {e}")
    sys.exit(1)

try:
    strain_L1_white = safe_whiten(strain_L1, L1_LABEL)
    print("Applying 30 Hz high-pass filter to L1...")
    strain_L1_hp = strain_L1_white.highpass(30)
    print("Applying 250 Hz low-pass filter to L1...")
    strain_L1_proc = strain_L1_hp.lowpass(250)
    print("Livingston (L1) preprocessing complete.")
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
    print("Slicing H1 data to 2-second window...")
    strain_H1_plot = strain_H1_proc.crop(plot_start, plot_end)
    print("Slicing L1 data to 2-second window...")
    strain_L1_plot = strain_L1_proc.crop(plot_start, plot_end)
except Exception as e:
    print(f"ERROR: Failed to crop data for plotting: {e}")
    sys.exit(1)

# Prepare time arrays relative to merger time
time_H1 = strain_H1_plot.times.value - GW170817_GPS
time_L1 = strain_L1_plot.times.value - GW170817_GPS

try:
    plt.figure(figsize=(10, 6))
    plt.plot(time_H1, strain_H1_plot.value, label=H1_LABEL, color='C0', alpha=0.8)
    plt.plot(time_L1, strain_L1_plot.value, label=L1_LABEL, color='C1', alpha=0.8)
    plt.axvline(0, color='k', linestyle='--', label='Merger Time')
    plt.xlabel('Time (s) relative to merger')
    plt.ylabel('Whitened, bandpassed strain')
    plt.title('GW170817: Whitened and Bandpassed Strain Data\nLIGO Hanford (H1) and Livingston (L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(STRAIN_PLOT_FILE, dpi=150)
    plt.show()
    print(f"Strain data visualization complete. Figure saved to {STRAIN_PLOT_FILE}")
except Exception as e:
    print(f"ERROR: Failed to plot strain time series: {e}")
    sys.exit(1)

# ==========================
# 4. Q-transform Spectrograms
# ==========================
print("="*60)
print("Step 4: Generating Q-transform spectrograms for both detectors...")

q_start = GW170817_GPS - PLOT_WINDOW_SEC
q_end = GW170817_GPS + PLOT_WINDOW_SEC

try:
    print("Cropping H1 data for Q-transform...")
    strain_H1_q = strain_H1_proc.crop(q_start, q_end)
    print("Cropping L1 data for Q-transform...")
    strain_L1_q = strain_L1_proc.crop(q_start, q_end)
except Exception as e:
    print(f"ERROR: Failed to crop data for Q-transform: {e}")
    sys.exit(1)

try:
    print("Computing Q-transform for H1...")
    qspec_H1 = strain_H1_q.q_transform(qrange=QTRANSFORM_QRANGE, frange=QTRANSFORM_FRANGE)
    print("Q-transform for H1 complete.")
except Exception as e:
    print(f"ERROR: Failed to compute Q-transform for H1: {e}")
    sys.exit(1)

try:
    print("Computing Q-transform for L1...")
    qspec_L1 = strain_L1_q.q_transform(qrange=QTRANSFORM_QRANGE, frange=QTRANSFORM_FRANGE)
    print("Q-transform for L1 complete.")
except Exception as e:
    print(f"ERROR: Failed to compute Q-transform for L1: {e}")
    sys.exit(1)

try:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # H1 spectrogram
    print("Plotting H1 Q-transform spectrogram...")
    qspec_H1.plot(
        ax=axes[0],
        vmin=0, vmax=15,
        cmap='viridis',
        norm=None,
        colorbar=True,
        rasterized=True
    )
    axes[0].axvline(GW170817_GPS, color='w', linestyle='--', label='Merger Time')
    axes[0].set_title(f'{H1_LABEL} Q-transform Spectrogram')
    axes[0].set_ylabel('Frequency [Hz]')
    axes[0].legend(loc='upper right')

    # L1 spectrogram
    print("Plotting L1 Q-transform spectrogram...")
    qspec_L1.plot(
        ax=axes[1],
        vmin=0, vmax=15,
        cmap='viridis',
        norm=None,
        colorbar=True,
        rasterized=True
    )
    axes[1].axvline(GW170817_GPS, color='w', linestyle='--', label='Merger Time')
    axes[1].set_title(f'{L1_LABEL} Q-transform Spectrogram')
    axes[1].set_xlabel('Time [GPS]')
    axes[1].set_ylabel('Frequency [Hz]')
    axes[1].legend(loc='upper right')

    # Adjust x-axis to show time relative to merger
    for ax in axes:
        xticks = ax.get_xticks()
        ax.set_xticklabels([f"{x - GW170817_GPS:.2f}" for x in xticks])
    axes[1].set_xlabel('Time (s) relative to merger')

    plt.tight_layout()
    plt.savefig(QTRANSFORM_PLOT_FILE, dpi=150)
    plt.show()
    print(f"Q-transform spectrogram visualization complete. Figure saved to {QTRANSFORM_PLOT_FILE}")
except Exception as e:
    print(f"ERROR: Failed to plot Q-transform spectrograms: {e}")
    sys.exit(1)

print("="*60)
print("All steps completed successfully.")