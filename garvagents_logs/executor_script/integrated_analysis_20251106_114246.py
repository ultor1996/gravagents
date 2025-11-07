# GW170817 Time-Frequency Analysis: 2s window, whitening, highpass 30 Hz, lowpass 250 Hz

import os
import sys
import traceback

from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# ========== PARAMETERS ==========
MERGER_GPS = 1187008882.4
WINDOW = 1.0  # seconds above and below merger
GPS_START = MERGER_GPS - WINDOW
GPS_END = MERGER_GPS + WINDOW
DETECTORS = ['H1', 'L1']

# Preprocessing
HIGHPASS_FREQ = 30
LOWPASS_FREQ = 250

# Q-transform
Q_QRANGE = (8, 64)
Q_FRANGE = (HIGHPASS_FREQ, LOWPASS_FREQ)
Q_OUTSEG = (0, 2)  # full 2-second window

# Output directories
OUTPUT_DIR = "gw170817_outputs"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
PROC_DIR = os.path.join(OUTPUT_DIR, "processed")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
for d in [OUTPUT_DIR, RAW_DIR, PROC_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 1. LOAD DATA ==========
print("\n=== [1/4] Loading GW170817 strain data (H1 & L1) ===")
strain_data = {}
for det in DETECTORS:
    print(f"  Loading strain data for {det} from GPS {GPS_START} to {GPS_END}...")
    try:
        ts = TimeSeries.fetch_open_data(det, GPS_START, GPS_END, cache=True)
        strain_data[det] = ts
        print(f"    Loaded {det}: {len(ts)} samples, sample rate {ts.sample_rate.value} Hz.")
        # Save raw data
        raw_path = os.path.join(RAW_DIR, f"{det}_raw.txt")
        ts.write(raw_path, format='txt')
        print(f"    Raw data saved to {raw_path}")
    except Exception as e:
        print(f"    ERROR: Could not load data for {det}: {e}")
        traceback.print_exc()
        strain_data[det] = None

# ========== 2. PREPROCESS DATA ==========
print("\n=== [2/4] Preprocessing strain data (whiten, highpass, lowpass) ===")
preprocessed_data = {}
for det in DETECTORS:
    ts = strain_data.get(det)
    if ts is None:
        print(f"  No raw data for {det}, skipping preprocessing.")
        preprocessed_data[det] = None
        continue
    try:
        print(f"  Preprocessing {det}...")
        ts_white = ts.whiten()
        ts_hp = ts_white.highpass(HIGHPASS_FREQ)
        ts_bp = ts_hp.lowpass(LOWPASS_FREQ)
        preprocessed_data[det] = ts_bp
        print(f"    {det} preprocessing complete: {len(ts_bp)} samples, sample rate {ts_bp.sample_rate.value} Hz.")
        # Save processed data
        proc_path = os.path.join(PROC_DIR, f"{det}_whitened_hp{HIGHPASS_FREQ}_lp{LOWPASS_FREQ}.txt")
        ts_bp.write(proc_path, format='txt')
        print(f"    Processed data saved to {proc_path}")
    except Exception as e:
        print(f"    ERROR: Preprocessing failed for {det}: {e}")
        traceback.print_exc()
        preprocessed_data[det] = None

# ========== 3. PLOT PROCESSED STRAIN ==========
print("\n=== [3/4] Plotting processed strain (2s window around merger) ===")
colors = {'H1': 'C0', 'L1': 'C1'}
labels = {'H1': 'Hanford (H1)', 'L1': 'Livingston (L1)'}
plotted = False

plt.figure(figsize=(10, 6))
for det in DETECTORS:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"  No preprocessed data for {det}, skipping plot.")
        continue
    try:
        times = ts.times.value - MERGER_GPS
        plt.plot(times, ts.value, label=labels[det], color=colors[det], alpha=0.8)
        plotted = True
        print(f"    Plotted {det} strain data.")
    except Exception as e:
        print(f"    ERROR: Could not plot {det}: {e}")
        traceback.print_exc()

if plotted:
    plt.xlabel("Time (s) relative to merger")
    plt.ylabel("Whitened strain")
    plt.title("GW170817: Whitened & Filtered Strain (±1s around merger)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    strain_plot_path = os.path.join(PLOT_DIR, "GW170817_strain_whitened_filtered.png")
    plt.savefig(strain_plot_path, dpi=150)
    print(f"  Strain plot saved to {strain_plot_path}")
    plt.show()
else:
    print("  No data was plotted. Please check preprocessing results.")

# ========== 4. Q-TRANSFORM SPECTROGRAMS ==========
print("\n=== [4/4] Generating Q-transform spectrograms ===")
for det in DETECTORS:
    ts = preprocessed_data.get(det)
    if ts is None:
        print(f"  No preprocessed data for {det}, skipping Q-transform.")
        continue
    try:
        print(f"  Computing Q-transform for {det}...")
        q = ts.q_transform(outseg=Q_OUTSEG, qrange=Q_QRANGE, frange=Q_FRANGE)
        print(f"    Q-transform computed. Plotting spectrogram...")
        fig = q.plot(figsize=(10, 6), vmin=0, vmax=5)
        ax = fig.gca()
        t0 = ts.t0.value
        # Center x-axis on merger
        ax.set_xlabel("Time (s) relative to merger")
        # Set xlim to -1 to +1 seconds relative to merger
        ax.set_xlim((MERGER_GPS - t0) - 1, (MERGER_GPS - t0) + 1)
        # Adjust x-ticks to be relative to merger
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{labels[det]} Q-transform Spectrogram (GW170817)")
        plt.tight_layout()
        qplot_path = os.path.join(PLOT_DIR, f"{det}_Qtransform_spectrogram.png")
        plt.savefig(qplot_path, dpi=150)
        print(f"    Q-transform spectrogram saved to {qplot_path}")
        plt.show()
    except Exception as e:
        print(f"    ERROR: Q-transform failed for {det}: {e}")
        traceback.print_exc()

print("\n=== Analysis complete! ===")
print(f"All outputs saved in: {OUTPUT_DIR}")