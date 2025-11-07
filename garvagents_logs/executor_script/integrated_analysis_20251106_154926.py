# === GW150914 End-to-End Matched Filtering Analysis ===
# Requirements: gwpy, pycbc, numpy, h5py, scipy, matplotlib (for plotting if desired)
# Author: Integrated Executor Agent

import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter
import os
import traceback

# ---------------------- PARAMETERS ----------------------
gw150914_gps = 1126259462.4
duration = 12  # seconds
detectors = ['H1', 'L1']
f_low = 30
f_high = 250
whiten_data = True  # Set to False to disable whitening
mass_range = range(20, 31)  # 20 to 30 inclusive
template_crop_duration = 1.0  # seconds
snr_crop_sec = 0.2  # seconds to crop from each edge

# Output directory
output_dir = "gw150914_matched_filter_results"
os.makedirs(output_dir, exist_ok=True)

# ---------------------- TASK 1: DATA LOADING & PREPROCESSING ----------------------
print("\n=== Task 1: Data Loading & Preprocessing ===")
gwpy_timeseries = {}
pycbc_timeseries = {}
sample_rates = {}
n_data = {}

for det in detectors:
    try:
        print(f"\nDownloading {duration}s of data for {det} centered on GW150914 (GPS {gw150914_gps})...")
        start = gw150914_gps - duration / 2
        end = gw150914_gps + duration / 2
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"Downloaded {det} data: {len(ts)} samples, sample rate {ts.sample_rate.value} Hz")

        print(f"Applying {f_low}-{f_high} Hz bandpass filter to {det} data...")
        ts_filtered = ts.bandpass(f_low, f_high)
        print("Bandpass filter applied.")

        if whiten_data:
            print(f"Whitening {det} data...")
            ts_filtered = ts_filtered.whiten()
            print("Whitening complete.")

        # Record sample rate and n_data
        sample_rates[det] = ts_filtered.sample_rate.value
        n_data[det] = len(ts_filtered)
        gwpy_timeseries[det] = ts_filtered

        # Convert to PyCBC TimeSeries
        print(f"Converting {det} GWpy TimeSeries to PyCBC TimeSeries...")
        pycbc_ts = PyCBC_TimeSeries(
            ts_filtered.value,
            delta_t=ts_filtered.dt.value,
            epoch=ts_filtered.t0.value
        )
        pycbc_timeseries[det] = pycbc_ts
        print(f"Conversion complete for {det}. delta_t: {pycbc_ts.delta_t}, epoch: {pycbc_ts.start_time}")

        # Save preprocessed data for reproducibility
        np.save(os.path.join(output_dir, f"{det}_preprocessed.npy"), ts_filtered.value)
    except Exception as e:
        print(f"Error processing {det}: {e}")
        traceback.print_exc()

# Save metadata
np.savez(os.path.join(output_dir, "data_metadata.npz"),
         sample_rates=sample_rates, n_data=n_data)

# ---------------------- TASK 2: TEMPLATE GENERATION & ALIGNMENT ----------------------
print("\n=== Task 2: Template Generation & Alignment ===")
# Use H1's parameters for template length and delta_t (assume H1 and L1 are identical for this analysis)
det_ref = 'H1'
if det_ref not in pycbc_timeseries:
    raise RuntimeError("Reference detector H1 data not available. Aborting.")

n_samples = n_data[det_ref]
delta_t = pycbc_timeseries[det_ref].delta_t
epoch = pycbc_timeseries[det_ref].start_time

templates = {}

for m1 in mass_range:
    for m2 in mass_range:
        if m1 < m2:
            continue  # Only do m1 >= m2 to avoid duplicates
        try:
            print(f"Generating template for m1={m1} M☉, m2={m2} M☉...")
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=f_low)
            # Crop to last 1s if duration > 1s
            if hp.duration > template_crop_duration:
                print(f"  Cropping template from {hp.duration:.2f}s to {template_crop_duration}s")
                hp = hp.time_slice(hp.end_time - template_crop_duration, hp.end_time)
            # Pad or truncate to n_samples
            if len(hp) < n_samples:
                print(f"  Padding template from {len(hp)} to {n_samples} samples")
                pad = np.zeros(n_samples - len(hp))
                hp_padded = np.concatenate([pad, hp.numpy()])
            else:
                print(f"  Truncating template from {len(hp)} to {n_samples} samples")
                hp_padded = hp.numpy()[-n_samples:]
            # Create new TimeSeries with correct epoch and delta_t
            template_ts = PyCBC_TimeSeries(hp_padded, delta_t=delta_t, epoch=epoch)
            templates[(m1, m2)] = template_ts
        except Exception as e:
            print(f"  Error generating template for m1={m1}, m2={m2}: {e}")
            traceback.print_exc()

# Save templates to HDF5 for reproducibility
with h5py.File(os.path.join(output_dir, "templates.h5"), "w") as f:
    for (m1, m2), ts in templates.items():
        dset_name = f"m1_{m1}_m2_{m2}"
        f.create_dataset(dset_name, data=ts.numpy())
    f.attrs['delta_t'] = float(delta_t)
    f.attrs['epoch'] = float(epoch)
    f.attrs['n_samples'] = int(n_samples)

# ---------------------- TASK 3: PSD ESTIMATION ----------------------
print("\n=== Task 3: PSD Estimation ===")
psds = {}

for det in pycbc_timeseries:
    try:
        print(f"\nEstimating PSD for {det}...")
        ts = pycbc_timeseries[det]
        sample_rate = sample_rates[det]
        n = n_data[det]
        seg_len_sec = 2.0
        seg_len = int(seg_len_sec * sample_rate)
        # Adjust seg_len as needed
        while (seg_len > n or seg_len < 32) and seg_len_sec > 0.01:
            seg_len_sec /= 2.0
            seg_len = int(seg_len_sec * sample_rate)
            print(f"  Adjusting seg_len_sec to {seg_len_sec:.3f}s ({seg_len} samples)")
        if seg_len > n or seg_len < 32:
            seg_len = n // 4
            print(f"  Fallback: using seg_len = n_data // 4 = {seg_len}")
        print(f"  Using seg_len = {seg_len} samples ({seg_len/sample_rate:.3f} s)")
        psd = welch(ts, seg_len=seg_len, avg_method='median')
        psds[det] = psd
        print(f"  PSD estimation complete for {det}. PSD length: {len(psd)}")
        # Save PSD to disk
        np.save(os.path.join(output_dir, f"{det}_psd.npy"), psd.numpy())
    except Exception as e:
        print(f"  Error estimating PSD for {det}: {e}")
        traceback.print_exc()

# ---------------------- TASK 4: MATCHED FILTERING & SNR EXTRACTION ----------------------
print("\n=== Task 4: Matched Filtering & SNR Extraction ===")
snr_results = {}  # (det, (m1, m2)) -> dict with max_abs_snr, max_abs_snr_time, snr_times, snr_values

for det in pycbc_timeseries:
    data = pycbc_timeseries[det]
    psd = psds[det]
    print(f"\nProcessing detector {det}...")
    for (m1, m2), template in templates.items():
        try:
            # Check length and delta_t
            if len(template) != len(data):
                print(f"  Skipping m1={m1}, m2={m2}: template/data length mismatch ({len(template)} vs {len(data)})")
                continue
            if template.delta_t != data.delta_t:
                print(f"  Skipping m1={m1}, m2={m2}: template/data delta_t mismatch ({template.delta_t} vs {data.delta_t})")
                continue

            print(f"  Matched filtering for template m1={m1}, m2={m2}...")
            snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=f_low)
            # Crop 0.2s from each edge
            crop_samples = int(snr_crop_sec / data.delta_t)
            if 2 * crop_samples >= len(snr):
                print(f"    Skipping: SNR series too short to crop {snr_crop_sec}s from each edge.")
                continue
            snr_cropped = snr.crop(crop_samples, crop_samples)
            # Find max absolute SNR and its time
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_abs_snr = abs_snr[max_idx]
            max_abs_snr_time = snr_cropped.sample_times[max_idx]
            # Store results
            snr_results[(det, (m1, m2))] = {
                'max_abs_snr': float(max_abs_snr),
                'max_abs_snr_time': float(max_abs_snr_time),
                'snr_times': snr_cropped.sample_times.numpy(),
                'snr_values': snr_cropped.numpy()
            }
            print(f"    Max |SNR| = {max_abs_snr:.2f} at t = {max_abs_snr_time:.4f} (GPS)")
        except Exception as e:
            print(f"    Error for m1={m1}, m2={m2}: {e}")
            traceback.print_exc()

# Save SNR results to HDF5
with h5py.File(os.path.join(output_dir, "snr_results.h5"), "w") as f:
    for (det, (m1, m2)), res in snr_results.items():
        grp = f.require_group(f"{det}/m1_{m1}_m2_{m2}")
        grp.create_dataset("snr_times", data=res['snr_times'])
        grp.create_dataset("snr_values", data=res['snr_values'])
        grp.attrs['max_abs_snr'] = res['max_abs_snr']
        grp.attrs['max_abs_snr_time'] = res['max_abs_snr_time']

print("\n=== Analysis Complete ===")
print(f"Results saved in directory: {output_dir}")