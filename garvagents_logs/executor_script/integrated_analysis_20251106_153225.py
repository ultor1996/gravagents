# =========================
# GW150914 Matched Filtering Analysis Pipeline
# =========================

# ---- Imports ----
import numpy as np
from pycbc.catalog import Merger
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter
import traceback
import sys
import os

# ---- Utility: Save results ----
def save_npz(filename, **kwargs):
    try:
        np.savez(filename, **kwargs)
        print(f"Saved results to {filename}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

# ---- Task 1: Data Loading ----
print("="*40)
print("TASK 1: Downloading GW150914 strain data")
print("="*40)
h1_strain = None
l1_strain = None
h1_sample_rate = None
l1_sample_rate = None
h1_num_samples = None
l1_num_samples = None
merger_time = None
start_time = None
end_time = None

try:
    print("Fetching GW150914 merger GPS time using PyCBC...")
    m = Merger('GW150914')
    merger_time = m.time  # GPS time of the merger
    print(f"Merger GPS time: {merger_time}")

    # Define time window: 8s before, 4s after
    start_time = merger_time - 8
    end_time = merger_time + 4
    print(f"Data window: {start_time} to {end_time} (GPS seconds)")

    # Download H1 data
    print("Downloading H1 strain data...")
    h1_strain = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    h1_sample_rate = h1_strain.sample_rate.value
    h1_num_samples = len(h1_strain)
    print(f"H1: Sample rate = {h1_sample_rate} Hz, Number of samples = {h1_num_samples}")

    # Download L1 data
    print("Downloading L1 strain data...")
    l1_strain = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    l1_sample_rate = l1_strain.sample_rate.value
    l1_num_samples = len(l1_strain)
    print(f"L1: Sample rate = {l1_sample_rate} Hz, Number of samples = {l1_num_samples}")

except Exception as e:
    print(f"Error during data loading: {e}")
    traceback.print_exc()
    sys.exit(1)

# Store results for later use
gw_data = {
    'H1': {
        'strain': h1_strain,
        'sample_rate': h1_sample_rate,
        'num_samples': h1_num_samples
    },
    'L1': {
        'strain': l1_strain,
        'sample_rate': l1_sample_rate,
        'num_samples': l1_num_samples
    },
    'merger_time': merger_time,
    'start_time': start_time,
    'end_time': end_time
}

# ---- Task 2: Preprocessing ----
print("\n" + "="*40)
print("TASK 2: Preprocessing (bandpass + whitening)")
print("="*40)
DO_WHITEN = True

try:
    print("Starting preprocessing of strain data...")

    # Retrieve strain data and metadata from previous task
    h1_strain = gw_data['H1']['strain']
    l1_strain = gw_data['L1']['strain']
    h1_sr = gw_data['H1']['sample_rate']
    l1_sr = gw_data['L1']['sample_rate']
    h1_len = gw_data['H1']['num_samples']
    l1_len = gw_data['L1']['num_samples']

    # Check sample rates and lengths
    print(f"H1 sample rate: {h1_sr} Hz, length: {h1_len}")
    print(f"L1 sample rate: {l1_sr} Hz, length: {l1_len}")

    # Resample if needed
    if h1_sr != l1_sr:
        print("Sample rates differ, resampling L1 to match H1...")
        l1_strain = l1_strain.resample(h1_sr)
        l1_sr = l1_strain.sample_rate.value

    # After resampling, check lengths
    if len(h1_strain) != len(l1_strain):
        min_len = min(len(h1_strain), len(l1_strain))
        print(f"Trimming both data streams to {min_len} samples for consistency...")
        h1_strain = h1_strain[:min_len]
        l1_strain = l1_strain[:min_len]

    # Apply bandpass filter (30–250 Hz)
    print("Applying 30–250 Hz bandpass filter to H1...")
    h1_bp = h1_strain.bandpass(30, 250, filtfilt=True)
    print("Applying 30–250 Hz bandpass filter to L1...")
    l1_bp = l1_strain.bandpass(30, 250, filtfilt=True)

    # Optionally whiten
    if DO_WHITEN:
        print("Whitening H1 data...")
        h1_proc = h1_bp.whiten()
        print("Whitening L1 data...")
        l1_proc = l1_bp.whiten()
    else:
        h1_proc = h1_bp
        l1_proc = l1_bp

    # Store results for later use
    processed_data = {
        'H1': h1_proc,
        'L1': l1_proc,
        'sample_rate': h1_proc.sample_rate.value,
        'num_samples': len(h1_proc)
    }
    print("Preprocessing complete.")

except Exception as e:
    print(f"Error during preprocessing: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---- Task 3: Template Generation ----
print("\n" + "="*40)
print("TASK 3: Generating waveform templates")
print("="*40)
data_length = processed_data['num_samples']
sample_rate = processed_data['sample_rate']
mass_range = range(20, 31)  # 20 to 30 M☉
templates = {}

print("Generating waveform templates for mass pairs 20–30 M☉ (zero spin)...")
for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates

        key = (m1, m2)
        try:
            # Generate waveform (PyCBC returns hp, hc)
            hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                     mass1=m1, mass2=m2,
                                     delta_t=1.0/sample_rate,
                                     f_lower=30.0,
                                     spin1z=0, spin2z=0)
            # Use only the plus polarization
            waveform = hp.data
            waveform_len = len(waveform)
            waveform_duration = waveform_len / sample_rate

            # Crop to last 1 second if longer than 1s
            if waveform_duration > 1.0:
                crop_samples = int(sample_rate)
                waveform = waveform[-crop_samples:]
            # If after cropping, waveform is too short, skip
            if len(waveform) < 0.5 * sample_rate:
                print(f"Template {(m1, m2)} too short after cropping, skipping.")
                continue

            # Pad or truncate to match data length
            if len(waveform) < data_length:
                pad_width = data_length - len(waveform)
                waveform = np.pad(waveform, (pad_width, 0), 'constant')
            elif len(waveform) > data_length:
                waveform = waveform[-data_length:]

            # Final check
            if len(waveform) != data_length:
                print(f"Template {(m1, m2)} length mismatch after processing, skipping.")
                continue

            templates[key] = waveform
            print(f"Template {(m1, m2)} generated and processed.")

        except Exception as e:
            print(f"Failed to generate template {(m1, m2)}: {e}")

print(f"Total valid templates generated: {len(templates)}")

# Optionally save templates for reproducibility
save_npz("templates.npz", **{str(k): v for k, v in templates.items()})

# ---- Task 4: PSD Estimation ----
print("\n" + "="*40)
print("TASK 4: Estimating PSDs")
print("="*40)
h1_proc = processed_data['H1']
l1_proc = processed_data['L1']
sample_rate = processed_data['sample_rate']
n_data = processed_data['num_samples']
psds = {}

def estimate_psd(strain, det_name):
    print(f"Estimating PSD for {det_name}...")
    seg_len_sec = 2.0
    seg_len_samples = int(seg_len_sec * sample_rate)
    min_seg_len_samples = 32
    max_attempts = 10
    attempts = 0

    # Adjust segment length as needed
    while (seg_len_samples > n_data or seg_len_samples < min_seg_len_samples) and attempts < max_attempts:
        seg_len_sec /= 2
        seg_len_samples = int(seg_len_sec * sample_rate)
        attempts += 1
        print(f"Adjusted segment length to {seg_len_sec:.3f} s ({seg_len_samples} samples)")

    # Fallback if still invalid
    if seg_len_samples > n_data or seg_len_samples < min_seg_len_samples:
        seg_len_samples = max(int(n_data // 4), min_seg_len_samples)
        print(f"Fallback: using segment length {seg_len_samples} samples")

    try:
        # Convert GWpy TimeSeries to numpy array if needed
        if hasattr(strain, 'value'):
            strain_data = strain.value
        else:
            strain_data = np.asarray(strain)
        # Estimate PSD
        psd = welch(strain_data, seg_len_samples, sample_rate, avg_method='median')
        print(f"PSD estimation for {det_name} complete. PSD length: {len(psd)}")
        return psd
    except Exception as e:
        print(f"Error estimating PSD for {det_name}: {e}")
        return None

# Estimate PSDs for both detectors
psds['H1'] = estimate_psd(h1_proc, 'H1')
psds['L1'] = estimate_psd(l1_proc, 'L1')

# Optionally save PSDs
try:
    np.savez("psds.npz", H1=psds['H1'], L1=psds['L1'])
    print("Saved PSDs to psds.npz")
except Exception as e:
    print(f"Failed to save PSDs: {e}")

# ---- Task 5: Matched Filtering ----
print("\n" + "="*40)
print("TASK 5: Matched Filtering")
print("="*40)
# Use H1 as the reference detector for this example
strain_data = processed_data['H1']
psd = psds['H1']
sample_rate = processed_data['sample_rate']
delta_t = 1.0 / sample_rate
n_samples = processed_data['num_samples']

results = {}

print("Starting matched filtering for all templates...")

# Convert strain to PyCBC TimeSeries if needed
if not isinstance(strain_data, PyCBC_TimeSeries):
    if hasattr(strain_data, 'value'):
        strain_array = strain_data.value
    else:
        strain_array = np.asarray(strain_data)
    strain_ts = PyCBC_TimeSeries(strain_array, delta_t=delta_t)
else:
    strain_ts = strain_data

for key, template in templates.items():
    try:
        # Convert template to PyCBC TimeSeries if needed
        if not isinstance(template, PyCBC_TimeSeries):
            template_ts = PyCBC_TimeSeries(template, delta_t=delta_t)
        else:
            template_ts = template

        # Ensure template and strain are the same length
        if len(template_ts) != len(strain_ts):
            min_len = min(len(template_ts), len(strain_ts))
            template_ts = template_ts[-min_len:]
            strain_ts = strain_ts[-min_len:]

        # Matched filtering
        snr = matched_filter(template_ts, strain_ts, psd=psd, low_frequency_cutoff=30.0)
        snr = snr.crop(0.2, 0.2)  # Crop 0.2 s from each edge

        # Find max absolute SNR and its time
        abs_snr = np.abs(snr)
        max_idx = np.argmax(abs_snr)
        max_snr = abs_snr[max_idx]
        max_time = snr.sample_times[max_idx]

        results[key] = {
            'max_abs_snr': float(max_snr),
            'max_time': float(max_time)
        }
        print(f"Template {key}: max |SNR| = {max_snr:.2f} at t = {max_time:.4f} s")

    except Exception as e:
        print(f"Matched filtering failed for template {key}: {e}")

print("Matched filtering complete.")

# Save results
try:
    # Save as a structured npy file
    np.save("matched_filter_results.npy", results)
    print("Saved matched filter results to matched_filter_results.npy")
except Exception as e:
    print(f"Failed to save matched filter results: {e}")

print("\nAll tasks completed successfully.")