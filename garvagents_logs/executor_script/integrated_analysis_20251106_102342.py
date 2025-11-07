# ============================================================
# GW170608 Gravitational Wave Data Analysis Script
# ============================================================
# This script downloads, filters, and analyzes strain data for
# GW170608 using GWpy and PyCBC, including time-domain and
# Q-transform visualizations.
# ============================================================

# ------------------------
# Imports
# ------------------------
from gwpy.timeseries import TimeSeries
from pycbc.catalog import Merger
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ------------------------
# Output Directory Setup
# ------------------------
OUTPUT_DIR = "gw170608_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Task 1: Data Loading
# ------------------------
print("="*60)
print("TASK 1: Downloading GW170608 strain data (H1, L1, ±32s)...")
print("="*60)
try:
    # Get GPS time for GW170608
    print("Fetching GPS time for GW170608...")
    merger = Merger('GW170608')
    event_time = float(merger.time)
    print(f"GW170608 GPS time: {event_time}")

    # Define time window
    duration = 64  # seconds
    half_window = duration // 2
    start_time = event_time - half_window
    end_time = event_time + half_window

    # Fetch strain data for H1 and L1
    detectors = ['H1', 'L1']
    strain_data = {}
    for det in detectors:
        try:
            print(f"Fetching {duration}s of strain data for {det} from {start_time} to {end_time}...")
            ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
            strain_data[det] = ts
            # Save raw data to disk
            ts.write(os.path.join(OUTPUT_DIR, f"strain_{det}_raw.gwf"), format='gwf')
            print(f"Successfully fetched and saved data for {det}.")
        except Exception as e:
            print(f"Error fetching data for {det}: {e}")
            strain_data[det] = None

    strain_H1 = strain_data['H1']
    strain_L1 = strain_data['L1']

    if strain_H1 is None or strain_L1 is None:
        print("ERROR: Failed to fetch strain data for one or both detectors. Exiting.")
        sys.exit(1)

except Exception as e:
    print(f"Critical error in Task 1: {e}")
    sys.exit(1)

# ------------------------
# Task 2: Filtering
# ------------------------
print("\n" + "="*60)
print("TASK 2: Applying 35-350 Hz bandpass filter to strain data...")
print("="*60)
try:
    # Define filter parameters
    low_freq = 35
    high_freq = 350

    filtered_strain = {}
    for det, strain in [('H1', strain_H1), ('L1', strain_L1)]:
        try:
            print(f"Applying {low_freq}-{high_freq} Hz bandpass filter to {det} data...")
            filtered = strain.bandpass(low_freq, high_freq)
            filtered_strain[det] = filtered
            # Save filtered data to disk
            filtered.write(os.path.join(OUTPUT_DIR, f"strain_{det}_filtered.gwf"), format='gwf')
            print(f"Filtering complete and saved for {det}.")
        except Exception as e:
            print(f"Error filtering {det} data: {e}")
            filtered_strain[det] = None

    filtered_strain_H1 = filtered_strain['H1']
    filtered_strain_L1 = filtered_strain['L1']

    if filtered_strain_H1 is None or filtered_strain_L1 is None:
        print("ERROR: Failed to filter strain data for one or both detectors. Exiting.")
        sys.exit(1)

except Exception as e:
    print(f"Critical error in Task 2: {e}")
    sys.exit(1)

# ------------------------
# Task 3: Time-Domain Plot
# ------------------------
print("\n" + "="*60)
print("TASK 3: Plotting time-domain filtered strain data...")
print("="*60)
try:
    # Define the zoom window (±0.25 seconds around the merger)
    window = 0.25  # seconds
    plot_start = event_time - window
    plot_end = event_time + window

    plt.figure(figsize=(12, 6))

    # Plot H1
    try:
        print("Plotting H1 filtered strain data...")
        ax1 = plt.subplot(2, 1, 1)
        filtered_strain_H1.plot(ax=ax1, color='C0', label='H1')
        ax1.set_xlim(plot_start, plot_end)
        ax1.set_ylabel('Strain')
        ax1.set_title('GW170608: H1 Filtered Strain (35-350 Hz)')
        ax1.axvline(event_time, color='k', linestyle='--', alpha=0.7, label='Merger Time')
        ax1.legend()
    except Exception as e:
        print(f"Error plotting H1 data: {e}")

    # Plot L1
    try:
        print("Plotting L1 filtered strain data...")
        ax2 = plt.subplot(2, 1, 2)
        filtered_strain_L1.plot(ax=ax2, color='C1', label='L1')
        ax2.set_xlim(plot_start, plot_end)
        ax2.set_xlabel('GPS Time (s)')
        ax2.set_ylabel('Strain')
        ax2.set_title('GW170608: L1 Filtered Strain (35-350 Hz)')
        ax2.axvline(event_time, color='k', linestyle='--', alpha=0.7, label='Merger Time')
        ax2.legend()
    except Exception as e:
        print(f"Error plotting L1 data: {e}")

    plt.tight_layout()
    time_domain_plot_path = os.path.join(OUTPUT_DIR, "filtered_strain_time_domain.png")
    plt.savefig(time_domain_plot_path)
    print(f"Time-domain plot saved to {time_domain_plot_path}")
    plt.show()
except Exception as e:
    print(f"Critical error in Task 3: {e}")

# ------------------------
# Task 4: Q-transform Spectrograms
# ------------------------
print("\n" + "="*60)
print("TASK 4: Generating Q-transform spectrograms...")
print("="*60)
try:
    # Define the Q-transform window (±1 second around the merger)
    q_window = 1.0  # seconds
    q_start = event_time - q_window
    q_end = event_time + q_window

    # Q-transform parameters
    q_kwargs = {
        'outseg': (q_start, q_end),
        'qrange': (8, 64),
        'frange': (20, 400),
        'logf': True,
        'stride': 0.01,
        'pad': 1,
    }

    # Generate and plot Q-transform for H1
    try:
        print("Generating Q-transform for H1...")
        qspec_H1 = filtered_strain_H1.q_transform(**q_kwargs)
        print("Plotting Q-transform for H1...")
        fig1 = qspec_H1.plot(figsize=(10, 4))
        ax1 = fig1.gca()
        ax1.set_title('GW170608: H1 Q-transform Spectrogram')
        ax1.axvline(event_time, color='w', linestyle='--', alpha=0.7, label='Merger Time')
        ax1.legend()
        plt.tight_layout()
        qtransform_h1_path = os.path.join(OUTPUT_DIR, "qtransform_H1.png")
        plt.savefig(qtransform_h1_path)
        print(f"H1 Q-transform spectrogram saved to {qtransform_h1_path}")
        plt.show()
    except Exception as e:
        print(f"Error generating or plotting Q-transform for H1: {e}")

    # Generate and plot Q-transform for L1
    try:
        print("Generating Q-transform for L1...")
        qspec_L1 = filtered_strain_L1.q_transform(**q_kwargs)
        print("Plotting Q-transform for L1...")
        fig2 = qspec_L1.plot(figsize=(10, 4))
        ax2 = fig2.gca()
        ax2.set_title('GW170608: L1 Q-transform Spectrogram')
        ax2.axvline(event_time, color='w', linestyle='--', alpha=0.7, label='Merger Time')
        ax2.legend()
        plt.tight_layout()
        qtransform_l1_path = os.path.join(OUTPUT_DIR, "qtransform_L1.png")
        plt.savefig(qtransform_l1_path)
        print(f"L1 Q-transform spectrogram saved to {qtransform_l1_path}")
        plt.show()
    except Exception as e:
        print(f"Error generating or plotting Q-transform for L1: {e}")

except Exception as e:
    print(f"Critical error in Task 4: {e}")

print("\nAnalysis complete. All outputs saved in:", OUTPUT_DIR)