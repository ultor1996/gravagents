# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
from gwpy.spectrum import FrequencySeries
import os
import traceback

# PyCBC imports
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries as PycbcTimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc import types

# --- Constants ---
EVENT_GPS = 1180922494.5
WINDOW = 32
GPS_START = EVENT_GPS - WINDOW
GPS_END = EVENT_GPS + WINDOW
DETECTORS = ['H1', 'L1']
MIN_SEGMENT_LENGTH = 60  # seconds

# --- Utility: Save results ---
def save_npz(filename, **kwargs):
    try:
        np.savez(filename, **kwargs)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"WARNING: Could not save {filename}: {e}")

def save_txt(filename, text):
    try:
        with open(filename, "w") as f:
            f.write(text)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"WARNING: Could not save {filename}: {e}")

# --- 1. Data Download & Validation ---
print("\n========== 1. DATA DOWNLOAD & VALIDATION ==========")
strain_data = {}
valid_segments = {}

for det in DETECTORS:
    print(f"\n--- Processing {det} ---")
    try:
        print(f"Fetching open data for {det} from {GPS_START} to {GPS_END}...")
        ts = TimeSeries.fetch_open_data(det, GPS_START, GPS_END, cache=True)
        print(f"Data fetched for {det}: {ts}")
    except Exception as e:
        print(f"ERROR: Failed to fetch data for {det}: {e}")
        strain_data[det] = None
        valid_segments[det] = []
        continue

    # Check for NaN or infinite values
    data = ts.value
    if np.any(np.isnan(data)):
        print(f"WARNING: NaN values found in {det} data.")
    if np.any(np.isinf(data)):
        print(f"WARNING: Infinite values found in {det} data.")

    # Remove NaN/infinite values for segment finding
    valid_mask = np.isfinite(data)
    if not np.all(valid_mask):
        print(f"Cleaning NaN/Inf from {det} data...")
        ts_clean = ts.copy()
        ts_clean.value[~valid_mask] = 0.0  # Replace with zeros
    else:
        ts_clean = ts

    # Find continuous valid segments of at least 60s
    print(f"Finding continuous segments of at least {MIN_SEGMENT_LENGTH}s for {det}...")
    dq_flag = DataQualityFlag(
        active=ts_clean.times[valid_mask],
        known=ts_clean.times,
        name=f'{det}_valid'
    )
    segments = dq_flag.active
    long_segments = [seg for seg in segments if seg.duration >= MIN_SEGMENT_LENGTH]
    if long_segments:
        print(f"Found {len(long_segments)} valid segment(s) >= {MIN_SEGMENT_LENGTH}s for {det}:")
        for seg in long_segments:
            print(f"  Segment: {seg}")
    else:
        print(f"ERROR: No valid {MIN_SEGMENT_LENGTH}s segment found for {det}.")

    strain_data[det] = ts
    valid_segments[det] = long_segments

# Final validation
for det in DETECTORS:
    if not valid_segments[det]:
        print(f"CRITICAL: {det} does not have a valid 60s segment. Analysis cannot proceed for this detector.")
    else:
        print(f"{det} is ready for further analysis.")

# Save raw data info
save_npz("strain_data.npz", **{det: strain_data[det].value for det in DETECTORS if strain_data[det] is not None})

# --- 2. Filtering & PSD Estimation ---
print("\n========== 2. FILTERING & PSD ESTIMATION ==========")
filtered_data = {}
psds = {}

def estimate_psd(ts, seglen, fftlength):
    try:
        print(f"  Estimating PSD: segment={seglen}s, fftlength={fftlength}...")
        psd = ts.psd(
            seglen=seglen,
            overlap=seglen // 2,
            window='hann',
            fftlength=fftlength,
            method='welch'
        )
        return psd
    except Exception as e:
        print(f"  PSD estimation failed: {e}")
        raise

def analytical_psd(ts):
    print("  Using analytical PSD approximation (flat noise)...")
    band = (ts.sample_rate.value // 2)
    freqs = np.linspace(0, band, len(ts))
    median_power = np.median(np.abs(ts.value)**2)
    psd_vals = np.ones_like(freqs) * median_power
    return FrequencySeries(psd_vals, frequencies=freqs, unit=ts.unit**2 / ts.sample_rate.unit)

for det in DETECTORS:
    print(f"\n--- Processing {det} ---")
    ts = strain_data.get(det)
    if ts is None:
        print(f"  No data for {det}, skipping.")
        filtered_data[det] = None
        psds[det] = None
        continue

    segments = valid_segments.get(det, [])
    if not segments:
        print(f"  No valid segment for {det}, skipping.")
        filtered_data[det] = None
        psds[det] = None
        continue
    seg = segments[0]
    print(f"  Using segment {seg} for filtering and PSD.")

    try:
        ts_seg = ts.crop(seg[0], seg[1])
        print(f"  Cropped data: {ts_seg}")
    except Exception as e:
        print(f"  ERROR: Cropping failed for {det}: {e}")
        filtered_data[det] = None
        psds[det] = None
        continue

    try:
        print("  Applying 30-300 Hz bandpass filter...")
        ts_filt = ts_seg.bandpass(30, 300)
        print("  Bandpass filter applied.")
    except Exception as e:
        print(f"  ERROR: Bandpass filtering failed for {det}: {e}")
        filtered_data[det] = None
        psds[det] = None
        continue

    filtered_data[det] = ts_filt

    psd = None
    for seglen, fftlength in [(4, 4096), (2, 4096)]:
        try:
            psd = estimate_psd(ts_filt, seglen, fftlength)
            if np.all(psd.value > 0):
                print(f"  PSD estimation successful for {det} with {seglen}s segments.")
                break
            else:
                print(f"  PSD for {det} contains non-positive values, trying fallback...")
                psd = None
        except Exception:
            continue

    if psd is None:
        try:
            psd = analytical_psd(ts_filt)
            if not np.all(psd.value > 0):
                print(f"  Analytical PSD for {det} still contains non-positive values!")
                psd = None
        except Exception as e:
            print(f"  ERROR: Analytical PSD failed for {det}: {e}")
            psd = None

    if psd is not None:
        print(f"  PSD ready for {det}.")
    else:
        print(f"  CRITICAL: No valid PSD for {det}.")

    psds[det] = psd

# Save filtered data and PSDs
for det in DETECTORS:
    if filtered_data[det] is not None:
        save_npz(f"filtered_data_{det}.npz", data=filtered_data[det].value)
    if psds[det] is not None:
        save_npz(f"psd_{det}.npz", psd=psds[det].value)

# --- 3. Whitening ---
print("\n========== 3. WHITENING ==========")
whitened_data = {}

for det in DETECTORS:
    print(f"\n--- Whitening {det} ---")
    ts_filt = filtered_data.get(det)
    psd = psds.get(det)

    if ts_filt is None:
        print(f"  No filtered data for {det}, skipping whitening.")
        whitened_data[det] = None
        continue

    if psd is None or not hasattr(psd, 'value') or not np.all(psd.value > 0):
        print(f"  PSD for {det} is invalid or missing. Skipping whitening, using filtered data only.")
        whitened_data[det] = ts_filt
        continue

    try:
        print(f"  Whitening {det} data using computed PSD...")
        ts_white = ts_filt.whiten(asd=psd**0.5)
        print(f"  Whitening successful for {det}.")
        whitened_data[det] = ts_white
    except Exception as e:
        print(f"  Whitening failed for {det}: {e}")
        print(f"  Proceeding with only filtered data for {det}.")
        whitened_data[det] = ts_filt

# Save whitened data
for det in DETECTORS:
    if whitened_data[det] is not None:
        save_npz(f"whitened_data_{det}.npz", data=whitened_data[det].value)

# --- 4. Template Bank & Matched Filtering ---
print("\n========== 4. TEMPLATE BANK & MATCHED FILTERING ==========")
primary_masses = np.arange(10, 15, 1)  # 10, 11, 12, 13, 14
secondary_masses = np.arange(7, 12, 1) # 7, 8, 9, 10, 11
spins = [-0.3, 0.0, 0.3]
approximant = "IMRPhenomPv2"
f_lower = 30
f_upper = 300
durations = [16, 8]  # Try 16s, fallback to 8s

pycbc_data = {}
pycbc_psds = {}

for det in DETECTORS:
    print(f"\n--- Preparing {det} data for PyCBC ---")
    ts = whitened_data.get(det)
    if ts is None:
        print(f"  No data for {det}, skipping.")
        pycbc_data[det] = None
        pycbc_psds[det] = None
        continue
    try:
        pycbc_ts = PycbcTimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)
        pycbc_data[det] = pycbc_ts
    except Exception as e:
        print(f"  ERROR: Could not convert {det} data to PyCBC TimeSeries: {e}")
        pycbc_data[det] = None
        pycbc_psds[det] = None
        continue

    psd = psds.get(det)
    if psd is not None and hasattr(psd, 'value') and np.all(psd.value > 0):
        try:
            print(f"  Interpolating PSD for {det}...")
            pycbc_psd = types.FrequencySeries(psd.value, delta_f=psd.df.value)
            pycbc_psd = interpolate(pycbc_psd, len(pycbc_ts)//2 + 1)
            pycbc_psd = inverse_spectrum_truncation(pycbc_psd, int(4 / ts.dt.value))
            pycbc_psds[det] = pycbc_psd
        except Exception as e:
            print(f"  PSD interpolation failed for {det}: {e}")
            pycbc_psds[det] = None
    else:
        print(f"  No valid PSD for {det}, using default PSD.")
        pycbc_psds[det] = None

results = []

print("\n--- Generating template bank and performing matched filtering ---")
for m1 in primary_masses:
    for m2 in secondary_masses:
        if m2 > m1:
            continue
        for spin1 in spins:
            for spin2 in spins:
                template_params = {
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1,
                    'spin2z': spin2,
                    'approximant': approximant,
                    'f_lower': f_lower,
                    'f_final': f_upper,
                }
                template = None
                for duration in durations:
                    try:
                        print(f"Generating template: m1={m1}, m2={m2}, spin1z={spin1}, spin2z={spin2}, duration={duration}s")
                        delta_t = None
                        for det in DETECTORS:
                            if pycbc_data[det] is not None:
                                delta_t = pycbc_data[det].delta_t
                                break
                        if delta_t is None:
                            delta_t = 1.0/4096
                        hp, _ = get_td_waveform(
                            **template_params,
                            delta_t=delta_t,
                            duration=duration
                        )
                        template = hp
                        break
                    except Exception as e:
                        print(f"  Template generation failed (duration={duration}s): {e}")
                        template = None
                if template is None:
                    print("  Skipping this parameter set due to template generation failure.")
                    continue

                snrs = {}
                for det in DETECTORS:
                    data = pycbc_data.get(det)
                    if data is None:
                        print(f"  No data for {det}, skipping matched filtering.")
                        snrs[det] = None
                        continue
                    try:
                        # Pad template to match data length
                        if len(template) > len(data):
                            template = template[:len(data)]
                        elif len(template) < len(data):
                            template = template.copy()
                            template.resize(len(data))
                        psd = pycbc_psds.get(det)
                        snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=f_lower)
                        peak_snr = abs(snr).numpy().max()
                        snrs[det] = peak_snr
                        print(f"    {det} peak SNR: {peak_snr:.2f}")
                    except Exception as e:
                        print(f"    Matched filtering failed for {det}: {e}")
                        snrs[det] = None

                if all(snrs[det] is not None for det in DETECTORS):
                    network_snr = np.sqrt(sum(snrs[det]**2 for det in DETECTORS))
                else:
                    network_snr = None

                results.append({
                    'params': template_params,
                    'snrs': snrs,
                    'network_snr': network_snr
                })

# Identify template with highest network SNR
valid_results = [r for r in results if r['network_snr'] is not None]
if valid_results:
    best_result = max(valid_results, key=lambda r: r['network_snr'])
    print("\n--- Peak template found ---")
    print(f"Parameters: {best_result['params']}")
    print(f"H1 SNR: {best_result['snrs']['H1']:.2f}, L1 SNR: {best_result['snrs']['L1']:.2f}")
    print(f"Network SNR: {best_result['network_snr']:.2f}")
    summary = (
        f"Best template parameters:\n{best_result['params']}\n"
        f"H1 SNR: {best_result['snrs']['H1']:.2f}\n"
        f"L1 SNR: {best_result['snrs']['L1']:.2f}\n"
        f"Network SNR: {best_result['network_snr']:.2f}\n"
    )
    save_txt("best_template_result.txt", summary)
else:
    best_result = None
    print("\nNo valid matched filter results found.")
    save_txt("best_template_result.txt", "No valid matched filter results found.")

# Save all results
try:
    import pickle
    with open("all_matched_filter_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved: all_matched_filter_results.pkl")
except Exception as e:
    print(f"WARNING: Could not save all_matched_filter_results.pkl: {e}")

print("\n========== PIPELINE COMPLETE ==========")