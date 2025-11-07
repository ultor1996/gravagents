# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter
import os
import sys
import traceback

# --- Parameters ---
merger_gps = 1126259462.4
start_time = merger_gps - 8  # 8 seconds before merger
end_time = merger_gps + 4    # 4 seconds after merger
detectors = ['H1', 'L1']
low_freq = 30
high_freq = 250
mass_range = range(20, 31)  # 20 to 30 inclusive
output_dir = "gw150914_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Data Loading ---
print("="*60)
print("TASK 1: Downloading GW150914 strain data for H1 and L1")
print("="*60)
strain_data = {}
sample_rates = {}
n_data_dict = {}

for det in detectors:
    print(f"Downloading strain data for {det} from {start_time} to {end_time}...")
    try:
        # FIX: Remove cache=True as per GWpy API change
        ts = TimeSeries.get(f'{det}:GWOSC-4KHZ_R1_STRAIN', start_time, end_time)
        strain_data[det] = ts
        sample_rates[det] = ts.sample_rate.value
        n_data_dict[det] = len(ts)
        print(f"Downloaded {det}: sample_rate = {sample_rates[det]} Hz, n_data = {n_data_dict[det]}")
        # Save raw data
        ts.write(os.path.join(output_dir, f"{det}_raw_strain.gwf"), format='gwf')
    except Exception as e:
        print(f"Error downloading data for {det}: {e}")
        traceback.print_exc()
        strain_data[det] = None
        sample_rates[det] = None
        n_data_dict[det] = None

if any(strain_data[det] is None for det in detectors):
    print("Critical error: Could not download data for all detectors. Exiting.")
    sys.exit(1)

# --- Task 2: Filtering & Conversion ---
print("\n" + "="*60)
print("TASK 2: Filtering strain data and converting to PyCBC TimeSeries")
print("="*60)
filtered_strain = {}
pycbc_strain = {}

def get_delta_t(ts):
    return 1.0 / ts.sample_rate.value

def get_epoch(ts):
    return ts.t0.value

for det in detectors:
    print(f"Applying {low_freq}-{high_freq} Hz bandpass filter to {det} data...")
    try:
        filtered = strain_data[det].bandpass(low_freq, high_freq)
        filtered_strain[det] = filtered
        print(f"Filtering complete for {det}.")
        # Save filtered data
        filtered.write(os.path.join(output_dir, f"{det}_filtered_strain.gwf"), format='gwf')
    except Exception as e:
        print(f"Error filtering {det}: {e}")
        traceback.print_exc()
        filtered_strain[det] = None

# Check and align delta_t and epoch
delta_t_H1 = get_delta_t(filtered_strain['H1'])
delta_t_L1 = get_delta_t(filtered_strain['L1'])
epoch_H1 = get_epoch(filtered_strain['H1'])
epoch_L1 = get_epoch(filtered_strain['L1'])

if not np.isclose(delta_t_H1, delta_t_L1, atol=1e-10):
    print("Sample rates differ, resampling L1 to match H1...")
    try:
        filtered_strain['L1'] = filtered_strain['L1'].resample(filtered_strain['H1'].sample_rate)
        print("Resampling complete.")
    except Exception as e:
        print(f"Error during resampling: {e}")
        traceback.print_exc()

# After resampling, check epochs
epoch_H1 = get_epoch(filtered_strain['H1'])
epoch_L1 = get_epoch(filtered_strain['L1'])

if not np.isclose(epoch_H1, epoch_L1, atol=1e-9):
    print(f"Epochs differ (H1: {epoch_H1}, L1: {epoch_L1}), aligning epochs...")
    new_epoch = max(epoch_H1, epoch_L1)
    end_time_aligned = min(filtered_strain['H1'].t1.value, filtered_strain['L1'].t1.value)
    for det in detectors:
        try:
            filtered_strain[det] = filtered_strain[det].crop(new_epoch, end_time_aligned)
            print(f"Cropped {det} to [{new_epoch}, {end_time_aligned}]")
        except Exception as e:
            print(f"Error cropping {det}: {e}")
            traceback.print_exc()

# Convert to PyCBC TimeSeries
for det in detectors:
    try:
        print(f"Converting {det} filtered GWpy TimeSeries to PyCBC TimeSeries...")
        pycbc_strain[det] = PyCBC_TimeSeries(filtered_strain[det].value,
                                             delta_t=get_delta_t(filtered_strain[det]),
                                             epoch=get_epoch(filtered_strain[det]))
        print(f"Conversion complete for {det}.")
        # Save PyCBC strain as numpy
        np.save(os.path.join(output_dir, f"{det}_pycbc_strain.npy"), pycbc_strain[det].numpy())
    except Exception as e:
        print(f"Error converting {det}: {e}")
        traceback.print_exc()
        pycbc_strain[det] = None

if any(pycbc_strain[det] is None for det in detectors):
    print("Critical error: Could not convert data for all detectors. Exiting.")
    sys.exit(1)

# Use H1 as reference for n_data and delta_t
delta_t = pycbc_strain['H1'].delta_t
n_data = len(pycbc_strain['H1'])

# --- Task 3: Template Generation & Conditioning ---
print("\n" + "="*60)
print("TASK 3: Generating and conditioning waveform templates")
print("="*60)
templates = {}

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates
        label = f"m1_{m1}_m2_{m2}"
        print(f"Generating template for m1={m1} M_sun, m2={m2} M_sun...")
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=delta_t,
                                    f_lower=30.0,
                                    spin1z=0, spin2z=0)
            # Crop if duration > 1 s
            if hp.duration > 1.0:
                print(f"Template duration {hp.duration:.2f}s > 1s, cropping to last 1s.")
                hp = hp.crop(hp.end_time - 1.0, hp.end_time)
            # Pad or truncate to match n_data
            if len(hp) < n_data:
                pad_width = n_data - len(hp)
                padded = np.pad(hp.numpy(), (pad_width, 0), 'constant')
                hp = PyCBC_TimeSeries(padded, delta_t=delta_t, epoch=hp.start_time - pad_width * delta_t)
                print(f"Padded template to {n_data} samples.")
            elif len(hp) > n_data:
                hp = hp[-n_data:]
                print(f"Truncated template to {n_data} samples.")
            # Ensure epoch matches data
            hp.start_time = pycbc_strain['H1'].start_time
            templates[label] = hp
            # Save template
            np.save(os.path.join(output_dir, f"template_{label}.npy"), hp.numpy())
            print(f"Template {label} ready: n_data={len(hp)}, delta_t={hp.delta_t}, epoch={hp.start_time}")
        except Exception as e:
            print(f"Error generating template {label}: {e}")
            traceback.print_exc()
            templates[label] = None

if not any(templates.values()):
    print("Critical error: No templates generated. Exiting.")
    sys.exit(1)

# --- Task 4: PSD Estimation & Matched Filtering ---
print("\n" + "="*60)
print("TASK 4: PSD estimation and matched filtering")
print("="*60)

def estimate_psd(data, delta_t, seg_len_sec=2.0):
    n_data = len(data)
    sample_rate = 1.0 / delta_t
    seg_len = int(seg_len_sec * sample_rate)
    while seg_len > n_data:
        seg_len = seg_len // 2
    if seg_len < 32:
        seg_len = max(32, n_data // 4)
    try:
        psd = welch(data, seg_len=seg_len, avg_method='median')
        print(f"PSD estimated with seg_len={seg_len} ({seg_len/sample_rate:.2f} s)")
        return psd
    except Exception as e:
        print(f"Error estimating PSD with seg_len={seg_len}: {e}")
        traceback.print_exc()
        seg_len = max(32, n_data // 4)
        try:
            psd = welch(data, seg_len=seg_len, avg_method='median')
            print(f"PSD fallback estimated with seg_len={seg_len}")
            return psd
        except Exception as e2:
            print(f"PSD estimation failed: {e2}")
            traceback.print_exc()
            return None

psds = {}
for det in detectors:
    print(f"Estimating PSD for {det}...")
    psds[det] = estimate_psd(pycbc_strain[det], delta_t)
    if psds[det] is not None:
        np.save(os.path.join(output_dir, f"{det}_psd.npy"), psds[det].numpy())

snr_results = {}

for det in detectors:
    data = pycbc_strain[det]
    psd = psds[det]
    if psd is None:
        print(f"Skipping {det} due to missing PSD.")
        continue
    snr_results[det] = {}
    for label, template in templates.items():
        if template is None:
            print(f"Skipping template {label} due to generation error.")
            continue
        print(f"Computing matched-filter SNR for {label} on {det}...")
        try:
            snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=30)
            crop_samples = int(0.2 / delta_t)
            if 2 * crop_samples >= len(snr):
                print(f"Warning: Not enough samples to crop 0.2s from each edge for {label} on {det}. Skipping.")
                continue
            snr_cropped = snr.crop(crop_samples, crop_samples)
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            snr_results[det][label] = {
                'max_abs_snr': float(max_snr),
                'max_time': float(max_time),
                # Save only the SNR series as numpy for space
                'snr_series_file': f"{det}_snr_{label}.npy"
            }
            np.save(os.path.join(output_dir, f"{det}_snr_{label}.npy"), snr_cropped.numpy())
            print(f"Max |SNR| for {label} on {det}: {max_snr:.2f} at t={max_time:.4f}")
        except Exception as e:
            print(f"Error computing SNR for {label} on {det}: {e}")
            traceback.print_exc()
            snr_results[det][label] = None

# Save SNR summary as a numpy file for easy loading
import json
with open(os.path.join(output_dir, "snr_results_summary.json"), "w") as f:
    json.dump(snr_results, f, indent=2)

print("\nAnalysis complete. Results saved in:", output_dir)