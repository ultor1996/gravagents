# =========================
# GW150914 Matched Filtering Analysis Pipeline
# =========================

# ---- Imports ----
import numpy as np
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import welch
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

def save_npy(filename, obj):
    try:
        np.save(filename, obj)
        print(f"Saved results to {filename}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

# ---- Task 1: Data Loading ----
print("="*40)
print("TASK 1: Downloading GW150914 strain data")
print("="*40)
gw150914_gps = 1126259462.4
duration = 12  # seconds
half_duration = duration / 2
start = gw150914_gps - half_duration
end = gw150914_gps + half_duration

gw_data = {}

for det in ['H1', 'L1']:
    print(f"Fetching {duration} s of {det} data centered on GW150914 (GPS {gw150914_gps})...")
    try:
        ts = GwpyTimeSeries.fetch_open_data(det, start, end, cache=True)
        sample_rate = ts.sample_rate.value if hasattr(ts.sample_rate, 'value') else float(ts.sample_rate)
        num_samples = len(ts)
        gw_data[det] = {
            'strain': ts,
            'sample_rate': sample_rate,
            'num_samples': num_samples
        }
        print(f"{det}: Data fetched. Sample rate = {sample_rate} Hz, Samples = {num_samples}")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        traceback.print_exc()
        sys.exit(1)

# Store sample rate and number of samples for later use
gw_data['sample_rate'] = gw_data['H1']['sample_rate'] if 'H1' in gw_data else None
gw_data['num_samples'] = gw_data['H1']['num_samples'] if 'H1' in gw_data else None

# ---- Task 2: Preprocessing ----
print("\n" + "="*40)
print("TASK 2: Preprocessing (bandpass + whitening)")
print("="*40)
DO_WHITEN = False  # Set to True to enable whitening

h1_ts = gw_data['H1']['strain']
l1_ts = gw_data['L1']['strain']

print("Checking sample rates and lengths for consistency...")
# Ensure sample rates match
if h1_ts.sample_rate != l1_ts.sample_rate:
    print(f"Sample rates differ: H1={h1_ts.sample_rate}, L1={l1_ts.sample_rate}. Resampling L1 to match H1.")
    l1_ts = l1_ts.resample(h1_ts.sample_rate)

# Ensure lengths match
min_len = min(len(h1_ts), len(l1_ts))
if len(h1_ts) != len(l1_ts):
    print(f"Lengths differ: H1={len(h1_ts)}, L1={len(l1_ts)}. Truncating to {min_len} samples.")
    h1_ts = h1_ts[:min_len]
    l1_ts = l1_ts[:min_len]

print("Applying 30–250 Hz bandpass filter to H1...")
try:
    h1_proc = h1_ts.bandpass(30, 250)
    print("H1 bandpass filtering complete.")
except Exception as e:
    print(f"Error bandpass filtering H1: {e}")
    h1_proc = None

print("Applying 30–250 Hz bandpass filter to L1...")
try:
    l1_proc = l1_ts.bandpass(30, 250)
    print("L1 bandpass filtering complete.")
except Exception as e:
    print(f"Error bandpass filtering L1: {e}")
    l1_proc = None

if DO_WHITEN:
    print("Applying whitening to H1...")
    try:
        h1_proc = h1_proc.whiten()
        print("H1 whitening complete.")
    except Exception as e:
        print(f"Error whitening H1: {e}")

    print("Applying whitening to L1...")
    try:
        l1_proc = l1_proc.whiten()
        print("L1 whitening complete.")
    except Exception as e:
        print(f"Error whitening L1: {e}")

processed_data = {
    'H1': h1_proc,
    'L1': l1_proc,
    'sample_rate': float(h1_ts.sample_rate.value if hasattr(h1_ts.sample_rate, 'value') else h1_ts.sample_rate),
    'num_samples': min_len
}

# ---- Task 3: Template Generation ----
print("\n" + "="*40)
print("TASK 3: Generating waveform templates")
print("="*40)
sample_rate = gw_data['sample_rate']
n_data = gw_data['num_samples']
delta_t = 1.0 / sample_rate
mass_range = range(20, 31)
templates = {}

print("Generating waveform templates for mass pairs 20–30 M☉ (zero spin)...")
for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only do m1 >= m2 to avoid duplicates
        key = (m1, m2)
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=delta_t,
                                    f_lower=30.0,
                                    spin1z=0, spin2z=0)
            # Crop if duration > 1 s
            if hp.duration > 1.0:
                crop_samples = int(1.0 / delta_t)
                hp = hp[-crop_samples:]
            # Pad or truncate to match n_data
            if len(hp) < n_data:
                pad = np.zeros(n_data - len(hp))
                hp = PyCBC_TimeSeries(np.concatenate([pad, hp.numpy()]), delta_t=delta_t)
            elif len(hp) > n_data:
                hp = hp[-n_data:]
            # Ensure final length matches n_data
            if len(hp) != n_data:
                print(f"Warning: Template {key} length mismatch after processing.")
            templates[key] = hp
            print(f"Template {key}: generated, length={len(hp)}")
        except Exception as e:
            print(f"Failed to generate template {key}: {e}")

print("Template generation complete.")
save_npz("templates.npz", **{str(k): v.numpy() for k, v in templates.items()})

# ---- Task 4: PSD Estimation ----
print("\n" + "="*40)
print("TASK 4: Estimating PSDs")
print("="*40)
h1_proc = processed_data['H1']
l1_proc = processed_data['L1']
sample_rate = processed_data['sample_rate']
n_data = processed_data['num_samples']

def compute_psd(strain, sample_rate, n_data, seg_len_sec=2.0, min_samples=32):
    seg_len = int(seg_len_sec * sample_rate)
    print(f"  Initial segment length: {seg_len} samples ({seg_len_sec} s)")
    # Adjust segment length as needed
    while (seg_len > n_data or seg_len < min_samples) and seg_len > 1:
        seg_len = seg_len // 2
        print(f"  Adjusted segment length: {seg_len} samples")
    if seg_len < min_samples or seg_len > n_data or seg_len < 1:
        seg_len = max(n_data // 4, min_samples)
        print(f"  Fallback segment length: {seg_len} samples")
    try:
        psd = welch(strain, seg_len=seg_len, avg_method='median')
        print(f"  PSD estimation successful (len={len(psd)})")
        return psd
    except Exception as e:
        print(f"  Error estimating PSD: {e}")
        return None

print("Estimating PSD for H1...")
h1_psd = compute_psd(h1_proc, sample_rate, n_data)

print("Estimating PSD for L1...")
l1_psd = compute_psd(l1_proc, sample_rate, n_data)

psd_dict = {
    'H1': h1_psd,
    'L1': l1_psd,
    'sample_rate': sample_rate,
    'n_data': n_data
}
save_npz("psds.npz", H1=h1_psd.numpy() if h1_psd is not None else None,
         L1=l1_psd.numpy() if l1_psd is not None else None)

# ---- Task 5: Matched Filtering ----
print("\n" + "="*40)
print("TASK 5: Matched Filtering")
print("="*40)
h1_strain = processed_data['H1']
l1_strain = processed_data['L1']
sample_rate = processed_data['sample_rate']
n_data = processed_data['num_samples']
h1_psd = psd_dict['H1']
l1_psd = psd_dict['L1']

# Convert GWpy TimeSeries to PyCBC TimeSeries if needed
def to_pycbc_timeseries(ts, sample_rate):
    if isinstance(ts, PyCBC_TimeSeries):
        return ts
    arr = ts.value if hasattr(ts, 'value') else np.asarray(ts)
    return PyCBC_TimeSeries(arr, delta_t=1.0/sample_rate)

print("Ensuring strain data are PyCBC TimeSeries with correct delta_t...")
h1_strain_pycbc = to_pycbc_timeseries(h1_strain, sample_rate)
l1_strain_pycbc = to_pycbc_timeseries(l1_strain, sample_rate)

def crop_edges(ts, crop_sec, sample_rate):
    crop_samples = int(crop_sec * sample_rate)
    if len(ts) < 2 * crop_samples:
        print("Warning: SNR series too short to crop requested edges.")
        return ts
    return ts[crop_samples:-crop_samples]

results = {'H1': {}, 'L1': {}}
crop_sec = 0.2

print("Starting matched filtering for all templates and detectors...")
for det, strain, psd in [('H1', h1_strain_pycbc, h1_psd), ('L1', l1_strain_pycbc, l1_psd)]:
    print(f"Processing detector {det}...")
    for key, template in templates.items():
        try:
            template_pycbc = to_pycbc_timeseries(template, sample_rate)
            if template_pycbc.delta_t != strain.delta_t:
                print(f"  Template {key} delta_t mismatch, resampling.")
                template_pycbc = PyCBC_TimeSeries(template_pycbc.numpy(), delta_t=strain.delta_t)
            snr = matched_filter(template_pycbc, strain, psd=psd, low_frequency_cutoff=30.0)
            snr_cropped = crop_edges(snr, crop_sec, sample_rate)
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            results[det][key] = {'max_abs_snr': float(max_snr), 'time': float(max_time)}
            print(f"  {det} template {key}: max|SNR|={max_snr:.2f} at t={max_time:.4f}")
        except Exception as e:
            print(f"  Error processing {det} template {key}: {e}")

print("Matched filtering complete.")
save_npy("matched_filter_results.npy", results)

print("\nAll tasks completed successfully.")