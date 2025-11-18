# --- Imports ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.types import TimeSeries
from scipy.signal import butter, filtfilt

# --- Section 1: Download GW150914 Strain Data ---
print("="*60)
print("Step 1: Downloading GW150914 strain data for Hanford (H1) and Livingston (L1)")
print("="*60)

strain_H1 = None
strain_L1 = None
gw_event = None

try:
    print("Fetching GW150914 event information from PyCBC catalog...")
    gw_event = Merger("GW150914")
    print(f"Event time (GPS): {gw_event.time}")

    # The PyCBC catalog provides a fixed segment around the event.
    # Do NOT specify start/end, just fetch the default segment.
    print("Downloading Hanford (H1) strain data...")
    strain_H1 = gw_event.strain('H1')
    if strain_H1 is None or len(strain_H1) == 0:
        raise RuntimeError("No Hanford (H1) strain data available for GW150914 in catalog segment.")

    print("Hanford (H1) strain data downloaded successfully.")

    print("Downloading Livingston (L1) strain data...")
    strain_L1 = gw_event.strain('L1')
    if strain_L1 is None or len(strain_L1) == 0:
        raise RuntimeError("No Livingston (L1) strain data available for GW150914 in catalog segment.")

    print("Livingston (L1) strain data downloaded successfully.")

    # Save raw strain data as numpy arrays for reproducibility
    np.save("strain_H1_raw.npy", strain_H1.numpy())
    np.save("strain_L1_raw.npy", strain_L1.numpy())
    print("Raw strain data saved to 'strain_H1_raw.npy' and 'strain_L1_raw.npy'.")

except Exception as e:
    print(f"Error occurred while downloading strain data: {e}", file=sys.stderr)
    print("Please ensure GW150914 strain data is available in the PyCBC catalog and do not specify custom time windows.", file=sys.stderr)
    sys.exit(1)

# --- Section 2: Bandpass Filtering ---
print("\n" + "="*60)
print("Step 2: Applying bandpass filter (35-350 Hz) to strain data")
print("="*60)

def bandpass_filter(strain, lowcut, highcut, order=4):
    """Apply a Butterworth bandpass filter to a PyCBC TimeSeries."""
    fs = float(strain.sample_rate)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, strain.numpy())
    return strain.__class__(filtered, delta_t=strain.delta_t, epoch=strain.start_time)

lowcut = 35.0
highcut = 350.0
order = 4

filtered_strain_H1 = None
filtered_strain_L1 = None

try:
    if strain_H1 is None or strain_L1 is None:
        raise ValueError("Strain data not found. Please check data loading step.")

    print("Applying bandpass filter to Hanford (H1) strain data...")
    filtered_strain_H1 = bandpass_filter(strain_H1, lowcut, highcut, order)
    print("Bandpass filter applied to Hanford (H1) data.")

    print("Applying bandpass filter to Livingston (L1) strain data...")
    filtered_strain_L1 = bandpass_filter(strain_L1, lowcut, highcut, order)
    print("Bandpass filter applied to Livingston (L1) data.")

    # Save filtered data as numpy arrays
    np.save("filtered_strain_H1.npy", filtered_strain_H1.numpy())
    np.save("filtered_strain_L1.npy", filtered_strain_L1.numpy())
    print("Filtered strain data saved to 'filtered_strain_H1.npy' and 'filtered_strain_L1.npy'.")

except Exception as e:
    print(f"Error during bandpass filtering: {e}", file=sys.stderr)
    sys.exit(1)

# --- Section 3: Visualization ---
print("\n" + "="*60)
print("Step 3: Plotting filtered strain vs time for H1 and L1")
print("="*60)

def plot_strain(strain, event_time, detector_label):
    """Plot filtered strain vs time relative to event."""
    times = strain.sample_times - event_time
    plt.plot(times, strain, label=detector_label)

try:
    if filtered_strain_H1 is None or filtered_strain_L1 is None:
        raise ValueError("Filtered strain data not found. Please check filtering step.")

    # Retrieve event time (GPS) from the catalog event object
    event_time = gw_event.time

    print("Plotting filtered strain data for H1 and L1...")

    plt.figure(figsize=(10, 6))
    plot_strain(filtered_strain_H1, event_time, 'Hanford (H1)')
    plot_strain(filtered_strain_L1, event_time, 'Livingston (L1)')

    plt.xlabel('Time (seconds, relative to event)')
    plt.ylabel('Strain (dimensionless)')
    plt.title('GW150914 Filtered Strain Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GW150914_filtered_strain.png")
    plt.show()
    print("Plot saved as 'GW150914_filtered_strain.png'.")
    print("Plotting complete.")

except Exception as e:
    print(f"Error during plotting: {e}", file=sys.stderr)
    sys.exit(1)