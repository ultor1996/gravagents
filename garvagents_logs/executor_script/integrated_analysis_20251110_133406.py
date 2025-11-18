# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries

# --- Constants and Parameters ---
MERGER_GPS = 1126259462.4
START_TIME = MERGER_GPS - 8  # 8 seconds before merger
END_TIME = MERGER_GPS + 4    # 4 seconds after merger
IFO_LIST = ['H1', 'L1']
BANDPASS_LOW = 20
BANDPASS_HIGH = 250
MASS_MIN = 10
MASS_MAX = 30
MIN_TEMPLATE_DURATION = 0.2  # seconds
OUTPUT_DIR = "gw_templates_output"

# --- Step 1: Download and Filter Data ---
print("="*60)
print("STEP 1: Downloading and filtering GW150914 strain data...")
strain_data = {}
filtered_strain_data = {}

for ifo in IFO_LIST:
    try:
        print(f"Fetching {ifo} strain data from {START_TIME} to {END_TIME} (GPS)...")
        strain = TimeSeries.fetch_open_data(ifo, START_TIME, END_TIME, cache=True)
        print(f"Successfully fetched {ifo} data. Applying {BANDPASS_LOW}-{BANDPASS_HIGH} Hz bandpass filter...")
        filtered_strain = strain.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print(f"Bandpass filter applied to {ifo} data.")
        strain_data[ifo] = strain
        filtered_strain_data[ifo] = filtered_strain
    except Exception as e:
        print(f"Error processing {ifo}: {e}")

if len(filtered_strain_data) != 2:
    raise RuntimeError("Failed to fetch and filter both H1 and L1 data. Aborting.")

# --- Step 2: Data Alignment ---
print("="*60)
print("STEP 2: Aligning H1 and L1 data (sample rate and length)...")
aligned_strain_data = {}

try:
    h1 = filtered_strain_data['H1']
    l1 = filtered_strain_data['L1']

    print("Checking sample rates (delta_t) and lengths...")
    h1_dt = h1.dt.value
    l1_dt = l1.dt.value
    h1_len = len(h1)
    l1_len = len(l1)
    print(f"H1: dt={h1_dt}, length={h1_len}")
    print(f"L1: dt={l1_dt}, length={l1_len}")

    # Step 1: Resample if needed
    if h1_dt != l1_dt:
        print("Sample rates differ. Resampling L1 to match H1...")
        l1 = l1.resample(1.0 / h1_dt)
        print(f"Resampled L1: dt={l1.dt.value}, length={len(l1)}")
    else:
        print("Sample rates are identical.")

    # Step 2: Trim to shortest length
    min_len = min(len(h1), len(l1))
    if len(h1) != len(l1):
        print(f"Lengths differ. Trimming both to {min_len} samples.")
    else:
        print("Lengths are identical.")

    h1_trimmed = h1[:min_len]
    l1_trimmed = l1[:min_len]

    aligned_strain_data['H1'] = h1_trimmed
    aligned_strain_data['L1'] = l1_trimmed

    print("Data alignment complete. Both H1 and L1 have:")
    print(f"  dt = {h1_trimmed.dt.value}")
    print(f"  length = {len(h1_trimmed)}")

except Exception as e:
    print(f"Error during data alignment: {e}")
    raise

# --- Step 3: Generate and Validate Templates ---
print("="*60)
print("STEP 3: Generating PyCBC waveform templates...")
ref_strain = aligned_strain_data['H1']
data_length = len(ref_strain)
delta_t = ref_strain.dt.value

mass_range = np.arange(MASS_MIN, MASS_MAX + 1)
templates = {}
template_params = []

n_total = 0
n_valid = 0

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only consider m2 <= m1 to avoid duplicates
        n_total += 1
        params = {
            'mass1': float(m1),
            'mass2': float(m2),
            'spin1z': 0.0,
            'spin2z': 0.0,
            'delta_t': delta_t,
            'f_lower': 20.0
        }
        try:
            hp, _ = get_td_waveform(**params)
            if hp is None or len(hp) == 0:
                print(f"Waveform generation failed for m1={m1}, m2={m2}")
                continue
            duration = len(hp) * delta_t
            if duration < MIN_TEMPLATE_DURATION:
                print(f"Template too short ({duration:.3f}s) for m1={m1}, m2={m2}, skipping.")
                continue
            # Pad or truncate to match data length
            if len(hp) < data_length:
                pad_width = data_length - len(hp)
                hp = hp.append_zeros(pad_width)
                print(f"Padded template for m1={m1}, m2={m2} to {data_length} samples.")
            elif len(hp) > data_length:
                hp = hp[:data_length]
                print(f"Truncated template for m1={m1}, m2={m2} to {data_length} samples.")
            # Store template and parameters
            templates[(m1, m2)] = hp
            template_params.append({'mass1': m1, 'mass2': m2, 'duration': duration})
            n_valid += 1
        except Exception as e:
            print(f"Error generating template for m1={m1}, m2={m2}: {e}")

print(f"Generated {n_valid} valid templates out of {n_total} mass pairs.")

if n_valid == 0:
    raise RuntimeError("No valid templates generated. Aborting.")

# Convert aligned strain data to PyCBC TimeSeries
pycbc_strain_data = {}
for ifo, gwpy_ts in aligned_strain_data.items():
    try:
        pycbc_ts = PyCBC_TimeSeries(gwpy_ts.value, delta_t=delta_t, epoch=gwpy_ts.t0.value)
        pycbc_strain_data[ifo] = pycbc_ts
        print(f"Converted {ifo} strain data to PyCBC TimeSeries.")
    except Exception as e:
        print(f"Error converting {ifo} strain data: {e}")
        raise

# --- Step 4: Visualization and Saving ---
print("="*60)
print("STEP 4: Overlaying templates and saving results...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save all templates as a 2D numpy array (n_templates x n_samples)
template_keys = list(templates.keys())
n_templates = len(template_keys)
n_samples = len(next(iter(templates.values())))
template_array = np.zeros((n_templates, n_samples), dtype=np.float64)
template_info = []

print("Saving template arrays to disk...")
for idx, key in enumerate(template_keys):
    template = templates[key]
    template_array[idx, :] = template.numpy()
    template_info.append({'m1': key[0], 'm2': key[1]})
np.save(os.path.join(OUTPUT_DIR, "templates.npy"), template_array)
np.save(os.path.join(OUTPUT_DIR, "template_info.npy"), template_info)
print(f"Saved {n_templates} templates and info to '{OUTPUT_DIR}'.")

# Overlay plots for each detector
for ifo, strain_ts in pycbc_strain_data.items():
    try:
        print(f"Plotting {ifo} strain data with overlaid templates...")
        plt.figure(figsize=(12, 6))
        t = strain_ts.sample_times.numpy()
        plt.plot(t, strain_ts.numpy(), label=f"{ifo} Filtered Strain", color='black', linewidth=1.2)
        # Overlay all templates (with alpha for visibility)
        for idx in range(n_templates):
            plt.plot(t, template_array[idx, :], alpha=0.2, color='C1')
        plt.title(f"{ifo} Strain Data with {n_templates} Overlaid Templates")
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"{ifo}_templates_overlay.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Error plotting for {ifo}: {e}")

print("All plots and template arrays saved successfully.")
print("="*60)
print("Workflow complete.")