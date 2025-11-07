ts = merger.strain(ifo)
# --- Imports ---
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.types import TimeSeries
from scipy.signal import butter, filtfilt

# --- Task 1: Download H1 and L1 strain data for GW150914 ---
print("="*60)
print("Starting download of GW150914 strain data for H1 and L1...")

event_name = "GW150914"
detectors = ['H1', 'L1']
strain_data = {}

try:
    print(f"Querying event information for {event_name}...")
    merger = Merger(event_name)
    event_time = merger.time
    print(f"Event GPS time: {event_time}")

    pad = 2  # seconds before and after event
    start = event_time - pad
    end = event_time + pad
    print(f"Downloading data from {start} to {end} (GPS seconds)")

    for det in detectors:
        print(f"Downloading strain data for {det}...")
        try:
            # FIX: Use det as positional argument, not detector=det
            ts = merger.strain(det, f_lower=20, sample_rate=4096, t1=start, t2=end)
            strain_data[det] = ts
            print(f"Successfully downloaded {det} data: {len(ts)} samples at {ts.sample_rate} Hz")
        except Exception as e:
            print(f"Error downloading data for {det}: {e}")
            strain_data[det] = None

except Exception as e:
    print(f"Failed to download GW150914 data: {e}")
    strain_data = None

if strain_data is None or any(strain_data[det] is None for det in detectors):
    print("Critical error: Could not download strain data for all detectors. Exiting.")
    sys.exit(1)

# --- Task 2: Whiten the H1 and L1 strain data ---
print("="*60)
print("Starting whitening of H1 and L1 strain data...")

whitened_data = {}

for det in detectors:
    ts = strain_data.get(det)
    if ts is None:
        print(f"No data available for {det}, skipping whitening.")
        whitened_data[det] = None
        continue

    try:
        print(f"Estimating PSD for {det}...")
        seglen = 2  # seconds
        psd = ts.psd(fftlength=seglen, overlap=seglen//2)
        psd = psd.interpolate(len(ts)//2 + 1)
        psd = psd.trim(ts.start_time, ts.end_time)
        print(f"PSD estimated for {det}.")

        print(f"Whitening {det} strain data...")
        whitened = ts.whiten(fftlength=seglen, overlap=seglen//2, psd=psd)
        whitened_data[det] = whitened
        print(f"Whitening complete for {det}.")

    except Exception as e:
        print(f"Error whitening data for {det}: {e}")
        whitened_data[det] = None

if any(whitened_data[det] is None for det in detectors):
    print("Critical error: Whitening failed for one or more detectors. Exiting.")
    sys.exit(1)

# --- Task 3: Apply bandpass filter (30-250 Hz) ---
print("="*60)
print("Starting bandpass filtering (30-250 Hz) of whitened data...")

filtered_data = {}

lowcut = 30.0
highcut = 250.0
order = 4  # Butterworth filter order

for det in detectors:
    ts = whitened_data.get(det)
    if ts is None:
        print(f"No whitened data available for {det}, skipping filtering.")
        filtered_data[det] = None
        continue

    try:
        print(f"Designing Butterworth bandpass filter for {det}...")
        fs = float(ts.sample_rate)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        print(f"Applying filter to {det} data...")
        filtered = filtfilt(b, a, ts.numpy())
        filtered_ts = TimeSeries(filtered, delta_t=ts.delta_t, epoch=ts.start_time)
        filtered_data[det] = filtered_ts
        print(f"Filtering complete for {det}.")

    except Exception as e:
        print(f"Error filtering data for {det}: {e}")
        filtered_data[det] = None

if any(filtered_data[det] is None for det in detectors):
    print("Critical error: Filtering failed for one or more detectors. Exiting.")
    sys.exit(1)

# --- Task 4: Plot the processed H1 and L1 strain data ---
print("="*60)
print("Preparing to plot processed H1 and L1 strain data...")

try:
    h1 = filtered_data.get('H1')
    l1 = filtered_data.get('L1')

    if h1 is None or l1 is None:
        raise ValueError("Filtered data for one or both detectors is missing.")

    # Use the time axis relative to the start time
    t0 = float(h1.start_time)
    times = np.arange(len(h1)) * h1.delta_t + t0
    times_rel = times - t0  # seconds relative to start

    plt.figure(figsize=(10, 6))
    plt.plot(times_rel, h1.numpy(), label='H1', color='C0', alpha=0.8)
    plt.plot(times_rel, l1.numpy(), label='L1', color='C1', alpha=0.8)
    plt.xlabel('Time (s) relative to segment start')
    plt.ylabel('Whitened, filtered strain')
    plt.title('GW150914: Whitened and Bandpass-Filtered Strain Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Plotting complete.")

except Exception as e:
    print(f"Error during plotting: {e}")

print("="*60)
print("Workflow complete.")