# =========================
# GW150914 Strain Data Download and Visualization
# =========================

import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger

# -------------------------
# Task 1: Download Strain Data
# -------------------------

event_name = "GW150914"
segment_duration = 4  # seconds
half_segment = segment_duration / 2

strain_data = {}
event_time = None

try:
    print(f"[INFO] Fetching event information for {event_name}...")
    merger = Merger(event_name)
    event_time = merger.time
    print(f"[INFO] Event time (GPS): {event_time}")

    # Define segment start and end
    segment_start = event_time - half_segment
    segment_end = event_time + half_segment

    # Use the robust Merger.strain(ifo) method as per PyCBC documentation
    for ifo in ['H1', 'L1']:
        print(f"[INFO] Downloading {segment_duration}s of strain data for {ifo} using Merger.strain()...")
        try:
            # Merger.strain() returns a TimeSeries object
            strain = merger.strain(ifo)
            # Slice the strain to the desired segment
            strain = strain.time_slice(segment_start, segment_end)
            strain_data[ifo] = strain
            print(f"[SUCCESS] Successfully downloaded strain data for {ifo}.")
            # Optionally save strain data to disk for reproducibility
            np.save(f"{event_name}_{ifo}_strain.npy", strain.numpy())
            np.save(f"{event_name}_{ifo}_times.npy", strain.sample_times.numpy())
        except Exception as e:
            print(f"[ERROR] Error downloading strain data for {ifo}: {e}", file=sys.stderr)
            strain_data[ifo] = None

except Exception as e:
    print(f"[FATAL] Failed to fetch event or strain data: {e}", file=sys.stderr)
    strain_data = {}

# -------------------------
# Task 2: Plot Strain vs Time
# -------------------------

try:
    print("[INFO] Preparing to plot strain data...")

    # Check that both detectors have data
    if strain_data.get('H1') is None or strain_data.get('L1') is None:
        raise ValueError("Strain data for one or both detectors is missing. Cannot plot.")

    # Prepare time arrays relative to event time
    h1 = strain_data['H1']
    l1 = strain_data['L1']
    t_h1 = h1.sample_times - event_time
    t_l1 = l1.sample_times - event_time

    plt.figure(figsize=(10, 6))
    plt.plot(t_h1, h1, label='Hanford (H1)', color='C0')
    plt.plot(t_l1, l1, label='Livingston (L1)', color='C1')
    plt.axvline(0, color='k', linestyle='--', label='Event time (GW150914)')
    plt.xlabel('Time (s) relative to event')
    plt.ylabel('Strain')
    plt.title('GW150914 Strain Time Series\nHanford (H1) and Livingston (L1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to disk for reproducibility
    plot_filename = f"{event_name}_strain_plot.png"
    plt.savefig(plot_filename, dpi=150)
    plt.show()
    print(f"[SUCCESS] Plot generated and saved as '{plot_filename}'.")

except Exception as e:
    print(f"[ERROR] Error during plotting: {e}", file=sys.stderr)
    sys.exit(1)