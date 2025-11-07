# --- Imports ---
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
EVENT_GPS = 1126259462.4
WINDOW = 2  # seconds before and after event
START_TIME = EVENT_GPS - WINDOW
END_TIME = EVENT_GPS + WINDOW

# Output file names for intermediate and final results
H1_RAW_FILE = "h1_strain.npy"
L1_RAW_FILE = "l1_strain.npy"
H1_WHITEN_FILE = "h1_whiten.npy"
L1_WHITEN_FILE = "l1_whiten.npy"
H1_FILTERED_FILE = "h1_filtered.npy"
L1_FILTERED_FILE = "l1_filtered.npy"

def save_timeseries(ts, filename):
    """Save GWpy TimeSeries to .npy file (time, value)."""
    if ts is not None:
        np.save(filename, np.vstack((ts.times.value, ts.value)))
        print(f"Saved {filename}")

def load_timeseries(filename):
    """Load (time, value) from .npy file."""
    if os.path.exists(filename):
        arr = np.load(filename)
        return arr[0], arr[1]
    else:
        return None, None

# --- 1. Download H1 and L1 strain data ---
print(f"Attempting to download H1 and L1 strain data for GW150914 from {START_TIME} to {END_TIME} (GPS)")

h1_strain = None
l1_strain = None

try:
    print("Fetching H1 strain data...")
    h1_strain = TimeSeries.fetch_open_data('H1', START_TIME, END_TIME, cache=True)
    print("H1 strain data downloaded successfully.")
    save_timeseries(h1_strain, H1_RAW_FILE)
except Exception as e:
    print(f"Error downloading H1 strain data: {e}")

try:
    print("Fetching L1 strain data...")
    l1_strain = TimeSeries.fetch_open_data('L1', START_TIME, END_TIME, cache=True)
    print("L1 strain data downloaded successfully.")
    save_timeseries(l1_strain, L1_RAW_FILE)
except Exception as e:
    print(f"Error downloading L1 strain data: {e}")

if h1_strain is None or l1_strain is None:
    print("Critical error: Could not download both H1 and L1 data. Exiting.")
    exit(1)

# --- 2. Whiten the strain data ---
print("Starting whitening of strain data...")

h1_whiten = None
l1_whiten = None

try:
    print("Whitening H1 strain data...")
    h1_whiten = h1_strain.whiten()
    print("H1 strain data whitened successfully.")
    save_timeseries(h1_whiten, H1_WHITEN_FILE)
except Exception as e:
    print(f"Error whitening H1 strain data: {e}")

try:
    print("Whitening L1 strain data...")
    l1_whiten = l1_strain.whiten()
    print("L1 strain data whitened successfully.")
    save_timeseries(l1_whiten, L1_WHITEN_FILE)
except Exception as e:
    print(f"Error whitening L1 strain data: {e}")

if h1_whiten is None or l1_whiten is None:
    print("Critical error: Could not whiten both H1 and L1 data. Exiting.")
    exit(1)

# --- 3. Apply lowpass and highpass filters ---
print("Applying lowpass (250 Hz) and highpass (30 Hz) filters to whitened data...")

h1_filtered = None
l1_filtered = None

try:
    print("Filtering H1 data...")
    h1_filtered = h1_whiten.lowpass(250).highpass(30)
    print("H1 data filtered successfully.")
    save_timeseries(h1_filtered, H1_FILTERED_FILE)
except Exception as e:
    print(f"Error filtering H1 data: {e}")

try:
    print("Filtering L1 data...")
    l1_filtered = l1_whiten.lowpass(250).highpass(30)
    print("L1 data filtered successfully.")
    save_timeseries(l1_filtered, L1_FILTERED_FILE)
except Exception as e:
    print(f"Error filtering L1 data: {e}")

if h1_filtered is None or l1_filtered is None:
    print("Critical error: Could not filter both H1 and L1 data. Exiting.")
    exit(1)

# --- 4. Plot the whitened and filtered strain data ---
print("Plotting whitened and filtered strain data for H1 and L1...")

try:
    plt.figure(figsize=(10, 6))
    # Plot H1
    plt.plot(h1_filtered.times.value, h1_filtered.value, label='H1', color='C0', alpha=0.7)
    # Plot L1
    plt.plot(l1_filtered.times.value, l1_filtered.value, label='L1', color='C1', alpha=0.7)
    plt.xlabel('Time (s) since GPS {:.1f}'.format(h1_filtered.times.value[0]))
    plt.ylabel('Whitened Strain')
    plt.title('Whitened and Filtered Strain Data: H1 vs L1 (GW150914)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Plotting complete.")
except Exception as e:
    print(f"Error during plotting: {e}")

print("Workflow complete. All intermediate and final results saved as .npy files.")