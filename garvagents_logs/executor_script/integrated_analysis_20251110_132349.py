# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform

# --- Parameters ---
merger_gps = 1126259462.4
start_time = merger_gps - 8  # 8 seconds before merger
end_time = merger_gps + 4    # 4 seconds after merger
detectors = ['H1', 'L1']
low_freq = 20
high_freq = 250

output_plot_dir = "overlay_plots"
output_template_dir = "template_arrays"
os.makedirs(output_plot_dir, exist_ok=True)
os.makedirs(output_template_dir, exist_ok=True)

# --- 1. Data Loading ---
print("="*60)
print("Step 1: Downloading GW150914 strain data for H1 and L1")
strain_data = {}
for det in detectors:
    try:
        print(f"Fetching data for {det} from {start_time} to {end_time} (GPS)...")
        # Remove deprecated 'cache' argument per database guidance
        strain = TimeSeries.fetch_open_data(det, start_time, end_time)
        strain_data[det] = strain
        print(f"Successfully fetched data for {det}.")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None

# Check if both detectors have data
if any(strain_data[det] is None for det in detectors):
    raise RuntimeError("Failed to fetch data for one or more detectors. Exiting.")

# --- 2. Filtering & Alignment ---
print("="*60)
print("Step 2: Filtering and aligning strain data")
filtered_data = {}
for det in detectors:
    try:
        print(f"Applying {low_freq}-{high_freq} Hz bandpass filter to {det}...")
        filtered = strain_data[det].bandpass(low_freq, high_freq)
        filtered_data[det] = filtered
        print(f"Filtering complete for {det}.")
    except Exception as e:
        print(f"Error filtering {det}: {e}")
        filtered_data[det] = None

# Check for successful filtering
if any(filtered_data[det] is None for det in detectors):
    raise RuntimeError("Filtering failed for one or more detectors. Exiting.")

# Ensure sample rates match
h1_rate = filtered_data['H1'].sample_rate.value
l1_rate = filtered_data['L1'].sample_rate.value
print(f"H1 sample rate: {h1_rate} Hz, L1 sample rate: {l1_rate} Hz")

if h1_rate != l1_rate:
    target_rate = min(h1_rate, l1_rate)
    print(f"Resampling both to {target_rate} Hz...")
    try:
        filtered_data['H1'] = filtered_data['H1'].resample(target_rate)
        filtered_data['L1'] = filtered_data['L1'].resample(target_rate)
        print("Resampling complete.")
    except Exception as e:
        raise RuntimeError(f"Error during resampling: {e}")

# Ensure lengths match
h1_len = len(filtered_data['H1'])
l1_len = len(filtered_data['L1'])
print(f"H1 length: {h1_len}, L1 length: {l1_len}")

min_len = min(h1_len, l1_len)
if h1_len != l1_len:
    print(f"Trimming both to {min_len} samples...")
    try:
        filtered_data['H1'] = filtered_data['H1'][:min_len]
        filtered_data['L1'] = filtered_data['L1'][:min_len]
        print("Trimming complete.")
    except Exception as e:
        raise RuntimeError(f"Error during trimming: {e}")

# --- 3. Template Generation ---
print("="*60)
print("Step 3: Generating PyCBC waveform templates")
data_ts = filtered_data['H1']
data_len = len(data_ts)
# Use .dt instead of .delta_t per GWpy documentation
delta_t = float(data_ts.dt)  # .dt is a float (seconds)
data_duration = data_len * delta_t
masses = np.arange(10, 31, 1)
templates = {}

for m1 in masses:
    for m2 in masses:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=delta_t,
                                    f_lower=20.0,
                                    spin1z=0, spin2z=0)
            if hp.duration < 0.2:
                continue
            waveform = hp.data
            wf_len = len(waveform)
            if wf_len < data_len:
                pad_width = data_len - wf_len
                waveform = np.pad(waveform, (pad_width, 0), 'constant')
            elif wf_len > data_len:
                waveform = waveform[-data_len:]
            templates[(m1, m2)] = waveform
            print(f"Template (m1={m1}, m2={m2}) generated, duration={hp.duration:.3f}s, length={len(waveform)}")
        except Exception as e:
            print(f"Error generating template for (m1={m1}, m2={m2}): {e}")

if len(templates) == 0:
    raise RuntimeError("No valid templates generated. Exiting.")
print(f"Total templates generated: {len(templates)}")

# --- 4. Overlay Visualization ---
print("="*60)
print("Step 4: Overlaying templates and saving plots/arrays")

def align_template_to_merger(template, data, merger_gps, data_start_gps, delta_t):
    """
    Shift template so its merger (peak) aligns with the merger time in the data.
    Assumes template is 1D numpy array, data is GWpy TimeSeries.
    """
    merger_idx_data = int(round((merger_gps - data_start_gps) / delta_t))
    merger_idx_template = np.argmax(np.abs(template))
    shift = merger_idx_data - merger_idx_template
    if shift > 0:
        aligned_template = np.pad(template, (shift, 0), 'constant')
        aligned_template = aligned_template[:len(data)]
    elif shift < 0:
        aligned_template = template[-shift:]
        if len(aligned_template) < len(data):
            aligned_template = np.pad(aligned_template, (0, len(data) - len(aligned_template)), 'constant')
        else:
            aligned_template = aligned_template[:len(data)]
    else:
        aligned_template = template[:len(data)]
    return aligned_template

for det in detectors:
    print(f"Processing overlays for {det}...")
    data = filtered_data[det]
    data_start_gps = data.times.value[0]
    # Use .dt instead of .delta_t
    delta_t = float(data.dt)
    data_len = len(data)
    time_axis = np.arange(data_len) * delta_t + data_start_gps
    for (m1, m2), template in templates.items():
        try:
            aligned_template = align_template_to_merger(template, data, merger_gps, data_start_gps, delta_t)
            # Save template array to HDF5
            template_fname = os.path.join(output_template_dir, f"template_{det}_m1_{m1}_m2_{m2}.h5")
            with h5py.File(template_fname, "w") as f:
                f.create_dataset("template", data=aligned_template)
                f.attrs['m1'] = m1
                f.attrs['m2'] = m2
                f.attrs['detector'] = det
                f.attrs['delta_t'] = delta_t
            # Plot overlay
            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, data.value, label=f"{det} strain", alpha=0.7)
            plt.plot(time_axis, aligned_template, label=f"Template m1={m1}, m2={m2}", alpha=0.7)
            plt.axvline(merger_gps, color='k', linestyle='--', label='Merger time')
            plt.xlabel("GPS Time (s)")
            plt.ylabel("Strain")
            plt.title(f"{det}: Overlay of Template (m1={m1}, m2={m2})")
            plt.legend()
            plt.tight_layout()
            plot_fname = os.path.join(output_plot_dir, f"overlay_{det}_m1_{m1}_m2_{m2}.png")
            plt.savefig(plot_fname)
            plt.close()
            print(f"Saved overlay plot and template for {det}, m1={m1}, m2={m2}")
        except Exception as e:
            print(f"Error processing overlay for {det}, m1={m1}, m2={m2}: {e}")

print("All overlays and templates saved.")
print("="*60)
print("Workflow complete.")