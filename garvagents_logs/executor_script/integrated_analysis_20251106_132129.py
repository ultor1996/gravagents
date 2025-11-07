# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.spectrum import FrequencySeries
import os
import traceback

# PyCBC imports
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries, FrequencySeries as PyCBC_FrequencySeries
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation

# --- Parameters ---
gps_center = 1180922494.5
duration = 64  # seconds
start = gps_center - duration / 2
end = gps_center + duration / 2
detectors = ['H1', 'L1']
min_segment_length = 60  # seconds

# --- Output directories ---
output_dir = "gw170608_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Data Loading and Validation ---
print("="*60)
print("STEP 1: Downloading and validating strain data")
print("="*60)

strain_data = {}
valid_segments = {}

for det in detectors:
    try:
        print(f"\nFetching data for {det}...")
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"Data fetched for {det}: {ts}")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        continue

    # Check for NaN or infinite values
    data = ts.value
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    bad_mask = nan_mask | inf_mask

    if np.any(bad_mask):
        print(f"Warning: {det} data contains NaN or infinite values. Cleaning...")
        ts_clean = ts.copy()
        ts_clean.value[bad_mask] = 0.0
    else:
        ts_clean = ts

    # Find continuous segments of valid data (no NaN/inf)
    valid = ~bad_mask
    min_samples = int(min_segment_length * ts.sample_rate.value)
    valid_indices = np.where(valid)[0]

    if valid_indices.size == 0:
        print(f"No valid data in {det}.")
        valid_segments[det] = []
        strain_data[det] = ts_clean
        continue

    splits = np.where(np.diff(valid_indices) != 1)[0] + 1
    segment_starts = np.insert(valid_indices[splits], 0, valid_indices[0])
    segment_ends = np.append(valid_indices[splits - 1], valid_indices[-1])

    segments = []
    for start_idx, end_idx in zip(segment_starts, segment_ends):
        seg_len = end_idx - start_idx + 1
        if seg_len >= min_samples:
            seg_start_time = ts_clean.times.value[start_idx]
            seg_end_time = ts_clean.times.value[end_idx]
            segments.append((seg_start_time, seg_end_time))

    if segments:
        print(f"{det}: Found {len(segments)} valid segment(s) of at least {min_segment_length}s.")
    else:
        print(f"{det}: No continuous segment of at least {min_segment_length}s found.")

    valid_segments[det] = segments
    strain_data[det] = ts_clean

proceed = any(len(segments) > 0 for segments in valid_segments.values())
if not proceed:
    print("\nERROR: No detector has a continuous 60s segment of valid data. Exiting.")
    exit(1)
else:
    print("\nSUCCESS: At least one detector has a valid 60s segment. Data is ready for further analysis.")

# Save cleaned data and segments
np.savez(os.path.join(output_dir, "strain_data.npz"), **{k: v.value for k, v in strain_data.items()})
np.savez(os.path.join(output_dir, "valid_segments.npz"), **valid_segments)

# --- 2. Filtering and PSD Estimation ---
print("\n" + "="*60)
print("STEP 2: Filtering and PSD estimation")
print("="*60)

filtered_data = {}
psds = {}

for det, ts in strain_data.items():
    print(f"\nProcessing {det}...")

    # 1. Bandpass filter
    try:
        print("  Applying bandpass filter (30-300 Hz)...")
        ts_bp = ts.bandpass(30, 300)
        filtered_data[det] = ts_bp
    except Exception as e:
        print(f"  ERROR: Bandpass filtering failed for {det}: {e}")
        continue

    # 2. PSD estimation (Welch, 4s segments, 50% overlap, hann, fftlength=4096)
    psd = None
    sample_rate = ts_bp.sample_rate.value
    fftlength = 4096  # samples

    try:
        print("  Estimating PSD (Welch, 4s segments)...")
        psd = ts_bp.psd(
            method='welch',
            fftlength=fftlength / sample_rate,
            overlap=0.5,
            window='hann'
        )
        if np.any(psd.value <= 0) or np.any(np.isnan(psd.value)) or np.any(np.isinf(psd.value)):
            raise ValueError("PSD contains non-positive, NaN, or infinite values.")
        print("  PSD estimation successful (4s segments).")
    except Exception as e:
        print(f"  WARNING: PSD estimation failed for {det} with 4s segments: {e}")
        try:
            print("  Trying PSD estimation with 2s segments...")
            nperseg_2s = int(2 * sample_rate)
            fftlength_2s = min(fftlength, nperseg_2s)
            psd = ts_bp.psd(
                method='welch',
                fftlength=fftlength_2s / sample_rate,
                overlap=0.5,
                window='hann'
            )
            if np.any(psd.value <= 0) or np.any(np.isnan(psd.value)) or np.any(np.isinf(psd.value)):
                raise ValueError("PSD contains non-positive, NaN, or infinite values.")
            print("  PSD estimation successful (2s segments).")
        except Exception as e2:
            print(f"  ERROR: PSD estimation failed for {det} with 2s segments: {e2}")
            try:
                print("  Using analytical (flat) PSD as fallback...")
                freqs = np.linspace(30, 300, 1000)
                flat_psd = np.ones_like(freqs) * np.var(ts_bp.value)
                psd = FrequencySeries(flat_psd, frequencies=freqs, unit=ts_bp.unit**2 / 'Hz')
                print("  Analytical PSD created.")
            except Exception as e3:
                print(f"  CRITICAL ERROR: Could not create analytical PSD for {det}: {e3}")
                psd = None

    if psd is not None and np.all(psd.value > 0) and not np.any(np.isnan(psd.value)) and not np.any(np.isinf(psd.value)):
        psds[det] = psd
        print(f"  PSD for {det} is valid and ready for use.")
    else:
        print(f"  ERROR: PSD for {det} is invalid. Skipping this detector.")
        filtered_data.pop(det, None)

# Save filtered data and PSDs
np.savez(os.path.join(output_dir, "filtered_data.npz"), **{k: v.value for k, v in filtered_data.items()})
for det, psd in psds.items():
    np.save(os.path.join(output_dir, f"psd_{det}.npy"), psd.value)

# --- 3. Whitening ---
print("\n" + "="*60)
print("STEP 3: Whitening")
print("="*60)

whitened_data = {}

for det in filtered_data:
    print(f"\nProcessing whitening for {det}...")
    ts = filtered_data[det]
    psd = psds.get(det, None)

    psd_valid = (
        psd is not None and
        hasattr(psd, 'value') and
        np.all(psd.value > 0) and
        not np.any(np.isnan(psd.value)) and
        not np.any(np.isinf(psd.value))
    )

    if not psd_valid:
        print(f"  WARNING: PSD for {det} is invalid or missing. Skipping whitening. Using filtered data.")
        whitened_data[det] = ts
        continue

    try:
        print("  Whitening data using computed PSD...")
        ts_white = ts.whiten(asd=np.sqrt(psd))
        whitened_data[det] = ts_white
        print("  Whitening successful.")
    except Exception as e:
        print(f"  ERROR: Whitening failed for {det}: {e}")
        print("  Proceeding with filtered (non-whitened) data.")
        whitened_data[det] = ts

# Save whitened data
np.savez(os.path.join(output_dir, "whitened_data.npz"), **{k: v.value for k, v in whitened_data.items()})

# --- 4. Template Generation and Matched Filtering ---
print("\n" + "="*60)
print("STEP 4: Template generation and matched filtering")
print("="*60)

# Template bank parameters
m1_vals = [10, 12, 14]
m2_vals = [7, 9, 11]
spin_vals = [-0.3, 0.0, 0.3]
approximant = "IMRPhenomPv2"
f_lower = 30
f_upper = 300
durations = [16, 8]  # Try 16s, fallback to 8s

# Convert gwpy TimeSeries to pycbc TimeSeries for each detector
pycbc_data = {}
pycbc_psds = {}
for det, ts in whitened_data.items():
    try:
        print(f"\nConverting {det} data to PyCBC TimeSeries...")
        data = np.array(ts.value, dtype=np.float64)
        delta_t = ts.dt.value
        pycbc_data[det] = PyCBC_TimeSeries(data, delta_t=delta_t)
        # Convert PSD to pycbc FrequencySeries and interpolate to match data length
        psd_freqs = psds[det].frequencies.value
        psd_vals = psds[det].value
        nfft = len(pycbc_data[det])
        df = 1.0 / (nfft * delta_t)
        interp_freqs = np.arange(0, 1.0/(2*delta_t), df)
        interp_psd_vals = np.interp(interp_freqs, psd_freqs, psd_vals, left=psd_vals[0], right=psd_vals[-1])
        pycbc_psds[det] = PyCBC_FrequencySeries(interp_psd_vals, delta_f=df)
    except Exception as e:
        print(f"  ERROR: Could not convert {det} data/PSD to PyCBC format: {e}")
        traceback.print_exc()
        pycbc_data.pop(det, None)
        pycbc_psds.pop(det, None)

if not pycbc_data:
    print("ERROR: No valid detectors for matched filtering. Exiting.")
    exit(1)

# Generate template bank
template_bank = []
for m1 in m1_vals:
    for m2 in m2_vals:
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                template_bank.append({
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1z,
                    'spin2z': spin2z
                })

print(f"\nGenerated {len(template_bank)} templates for the bank.")

network_snr_peaks = []
template_results = []

for idx, params in enumerate(template_bank):
    print(f"\n[{idx+1}/{len(template_bank)}] Generating template: {params}")
    waveform = None
    for duration in durations:
        try:
            hp, _ = get_td_waveform(approximant=approximant,
                                    mass1=params['mass1'],
                                    mass2=params['mass2'],
                                    spin1z=params['spin1z'],
                                    spin2z=params['spin2z'],
                                    delta_t=list(pycbc_data.values())[0].delta_t,
                                    f_lower=f_lower,
                                    f_final=f_upper,
                                    duration=duration)
            waveform = hp
            print(f"  Template generated with duration {duration}s.")
            break
        except Exception as e:
            print(f"  WARNING: Template generation failed for duration {duration}s: {e}")
    if waveform is None:
        print("  ERROR: Skipping template due to generation failure.")
        continue

    snr_dict = {}
    for det in pycbc_data:
        try:
            data = pycbc_data[det]
            psd = pycbc_psds[det]
            # Truncate or pad waveform to match data length
            if len(waveform) > len(data):
                waveform = waveform[:len(data)]
            elif len(waveform) < len(data):
                waveform = waveform.copy()
                waveform.resize(len(data))
            print(f"    Matched filtering on {det}...")
            snr = matched_filter(waveform, data, psd=psd, low_frequency_cutoff=f_lower)
            snr_dict[det] = snr
        except Exception as e:
            print(f"    ERROR: Matched filtering failed for {det}: {e}")
            snr_dict[det] = None

    peak_snrs = []
    for det, snr in snr_dict.items():
        if snr is not None:
            peak_snrs.append(np.max(np.abs(snr)))
    if peak_snrs:
        network_snr = np.sqrt(np.sum(np.square(peak_snrs)))
    else:
        network_snr = 0.0

    network_snr_peaks.append(network_snr)
    template_results.append({
        'params': params,
        'snr_dict': snr_dict,
        'network_snr': network_snr
    })
    print(f"  Network SNR for this template: {network_snr:.2f}")

# Identify peak template
if network_snr_peaks:
    best_idx = int(np.argmax(network_snr_peaks))
    best_template = template_results[best_idx]
    print("\nBest template parameters:", best_template['params'])
    print("Best network SNR:", best_template['network_snr'])
    # Save best template info
    with open(os.path.join(output_dir, "best_template.txt"), "w") as f:
        f.write(f"Best template parameters: {best_template['params']}\n")
        f.write(f"Best network SNR: {best_template['network_snr']}\n")
else:
    print("\nNo valid templates produced a network SNR.")

# Save all template results
np.save(os.path.join(output_dir, "template_results.npy"), template_results)
print(f"\nAll results saved in '{output_dir}'.")

print("\nPipeline complete.")