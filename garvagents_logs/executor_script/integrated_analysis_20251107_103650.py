# === Gravitational Wave Data Analysis Pipeline ===
# Tasks: Data Loading & Filtering -> PyCBC Conversion -> Template Generation -> PSD & Matched Filtering

import os
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter

# --- Utility Functions ---
def save_numpy_dict(filename, d):
    """Save a dictionary of numpy arrays or objects to a .npz file."""
    np.savez_compressed(filename, **d)

def save_max_snr_stats(filename, max_snr_stats):
    """Save max SNR stats as a CSV file."""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['detector', 'm1', 'm2', 'max_abs_snr', 'time'])
        for det in max_snr_stats:
            for (m1, m2), (max_snr, max_time) in max_snr_stats[det].items():
                writer.writerow([det, m1, m2, max_snr, max_time])

# --- Parameters ---
gps_start = 1126259462
gps_end = gps_start + 4  # 4 seconds of data
channels = {
    'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
    'L1': 'L1:GWOSC-4KHZ_R1_STRAIN'
}
detectors = ['H1', 'L1']
mass_range = range(20, 31)  # 20–30 M☉

# --- Output Directories ---
os.makedirs('results', exist_ok=True)

# === Task 1: Data Loading and Filtering ===
print("\n=== Task 1: Loading and Filtering Strain Data ===")
strain_data = {}
filtered_data = {}
sample_rates = {}
n_data = {}

for det in detectors:
    print(f"\nLoading raw strain data for {det}...")
    try:
        ts = TimeSeries.get(channels[det], gps_start, gps_end, cache=True)
        strain_data[det] = ts
        sample_rates[det] = ts.sample_rate.value
        n_data[det] = len(ts)
        print(f"{det}: Loaded {n_data[det]} samples at {sample_rates[det]} Hz.")
    except Exception as e:
        print(f"Error loading data for {det}: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        n_data[det] = None
        continue

    print(f"Applying 30–250 Hz Butterworth bandpass filter to {det} data...")
    try:
        filtered = ts.bandpass(30, 250)
        filtered_data[det] = filtered
        print(f"{det}: Filtering complete.")
    except Exception as e:
        print(f"Error filtering data for {det}: {e}")
        filtered_data[det] = None

# Save sample rates and n_data for reproducibility
np.savez('results/sample_rates_and_n_data.npz', sample_rates=sample_rates, n_data=n_data)

# === Task 2: Conversion to PyCBC TimeSeries ===
print("\n=== Task 2: Converting to PyCBC TimeSeries ===")
pycbc_timeseries = {}

for det in detectors:
    print(f"\nConverting filtered GWpy TimeSeries to PyCBC TimeSeries for {det}...")
    gwpy_ts = filtered_data.get(det)
    if gwpy_ts is None:
        print(f"Warning: No filtered data available for {det}. Skipping conversion.")
        pycbc_timeseries[det] = None
        continue

    try:
        data_array = gwpy_ts.value
        delta_t = 1.0 / sample_rates[det]
        epoch = float(gwpy_ts.t0.value)
        ts_pycbc = PyCBC_TimeSeries(data_array, delta_t=delta_t, epoch=epoch)
        pycbc_timeseries[det] = ts_pycbc
        print(f"{det}: Conversion successful. Length: {len(ts_pycbc)}, delta_t: {delta_t}, epoch: {epoch}")
    except Exception as e:
        print(f"Error converting {det} data to PyCBC TimeSeries: {e}")
        pycbc_timeseries[det] = None

# Save PyCBC TimeSeries arrays for reproducibility
for det in detectors:
    ts = pycbc_timeseries.get(det)
    if ts is not None:
        np.save(f'results/pycbc_timeseries_{det}.npy', ts.numpy())

# === Task 3: Template Generation and Alignment ===
print("\n=== Task 3: Generating and Aligning Templates ===")
ref_det = 'H1'
if pycbc_timeseries[ref_det] is None:
    raise RuntimeError("Reference detector data missing. Cannot proceed with template generation.")

delta_t = pycbc_timeseries[ref_det].delta_t
n_data_ref = len(pycbc_timeseries[ref_det])
templates = {}

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue
        print(f"Generating template for m1={m1} M☉, m2={m2} M☉...")
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=30.0)
            max_samples = int(1.0 / delta_t)
            if len(hp) > max_samples:
                print(f"  Cropping template from {len(hp)} to {max_samples} samples (1 s)")
                hp = hp[-max_samples:]
            if len(hp) < n_data_ref:
                pad_width = n_data_ref - len(hp)
                print(f"  Padding template with {pad_width} zeros at the start")
                hp = np.pad(hp, (pad_width, 0), 'constant')
            elif len(hp) > n_data_ref:
                print(f"  Truncating template from {len(hp)} to {n_data_ref} samples")
                hp = hp[-n_data_ref:]
            template_ts = PyCBC_TimeSeries(hp, delta_t=delta_t, epoch=0)
            templates[(m1, m2)] = template_ts
            print(f"  Template ready: {len(template_ts)} samples, delta_t={delta_t}")
        except Exception as e:
            print(f"  Error generating template for m1={m1}, m2={m2}: {e}")
            templates[(m1, m2)] = None

# Save templates for reproducibility
for (m1, m2), template in templates.items():
    if template is not None:
        np.save(f'results/template_m1_{m1}_m2_{m2}.npy', template.numpy())

# === Task 4: PSD Estimation and Matched Filtering ===
print("\n=== Task 4: PSD Estimation and Matched Filtering ===")
psds = {}
snr_results = {}
max_snr_stats = {}

for det in detectors:
    print(f"\nEstimating PSD for {det}...")
    data = pycbc_timeseries[det]
    if data is None:
        print(f"  No data for {det}, skipping PSD estimation and filtering.")
        psds[det] = None
        snr_results[det] = {}
        max_snr_stats[det] = {}
        continue

    seg_len_sec = 2.0
    n_data_det = len(data)
    delta_t = data.delta_t
    sample_rate = 1.0 / delta_t
    while True:
        seg_len = int(seg_len_sec * sample_rate)
        if seg_len < 32:
            seg_len = max(32, n_data_det // 4)
            print(f"  Segment length too short, using seg_len={seg_len} samples.")
            break
        if seg_len > n_data_det:
            seg_len_sec /= 2
            if seg_len_sec < delta_t * 32:
                seg_len = max(32, n_data_det // 4)
                print(f"  Segment length too long, using seg_len={seg_len} samples.")
                break
            continue
        break

    try:
        psd = welch(data, seg_len=seg_len, avg_method='median')
        psds[det] = psd
        print(f"  PSD estimated with seg_len={seg_len} samples ({seg_len/sample_rate:.2f} s).")
        np.save(f'results/psd_{det}.npy', psd.numpy())
    except Exception as e:
        print(f"  Error estimating PSD for {det}: {e}")
        psds[det] = None
        snr_results[det] = {}
        max_snr_stats[det] = {}
        continue

    snr_results[det] = {}
    max_snr_stats[det] = {}

    for (m1, m2), template in templates.items():
        if template is None:
            print(f"  Skipping template m1={m1}, m2={m2} (generation failed).")
            snr_results[det][(m1, m2)] = None
            max_snr_stats[det][(m1, m2)] = (None, None)
            continue
        print(f"  Matched filtering {det} data with template m1={m1}, m2={m2}...")
        try:
            snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=30)
            crop_samples = int(0.2 / delta_t)
            if 2 * crop_samples >= len(snr):
                print(f"    Warning: SNR series too short to crop 0.2 s from each edge.")
                snr_cropped = snr
            else:
                snr_cropped = snr[crop_samples:-crop_samples]
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            snr_results[det][(m1, m2)] = snr_cropped
            max_snr_stats[det][(m1, m2)] = (max_snr, max_time)
            print(f"    Max |SNR|: {max_snr:.2f} at t={max_time:.4f} s")
            # Save SNR time series for this template/detector
            np.save(f'results/snr_{det}_m1_{m1}_m2_{m2}.npy', snr_cropped.numpy())
        except Exception as e:
            print(f"    Error in matched filtering for m1={m1}, m2={m2}: {e}")
            snr_results[det][(m1, m2)] = None
            max_snr_stats[det][(m1, m2)] = (None, None)

# Save max SNR stats as CSV
save_max_snr_stats('results/max_snr_stats.csv', max_snr_stats)

print("\n=== Pipeline Complete ===")
print("All intermediate and final results saved in the 'results/' directory.")