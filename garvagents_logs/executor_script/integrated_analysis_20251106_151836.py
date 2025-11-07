# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.filter import highpass, lowpass, matched_filter
from pycbc.waveform import get_td_waveform
import sys
import traceback

# --- Parameters ---
center_gps = 1126259462.422
duration = 12  # seconds
half_duration = duration / 2
start_gps = center_gps - half_duration
end_gps = center_gps + half_duration
detectors = ['H1', 'L1']
bandpass = (30, 250)
whiten = True
mass_range = np.arange(20, 31)
crop_edge_sec = 0.2

# --- 1. Data Loading ---
print("="*60)
print("STEP 1: Downloading GW150914 strain data (12s window)...")
strain_data = {}
sample_rates = {}
num_samples = {}

for det in detectors:
    print(f"Fetching {duration} s of strain data for {det} from GPS {start_gps} to {end_gps}...")
    try:
        ts = GwpyTimeSeries.fetch_open_data(det, start_gps, end_gps, cache=True)
        strain_data[det] = ts
        sample_rates[det] = ts.sample_rate.value
        num_samples[det] = len(ts)
        print(f"  {det}: Success. Sample rate = {sample_rates[det]} Hz, Samples = {num_samples[det]}")
    except Exception as e:
        print(f"  {det}: ERROR fetching data: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        num_samples[det] = None

# Check that at least one detector succeeded
if all(strain_data[det] is None for det in detectors):
    print("ERROR: No strain data could be loaded. Exiting.")
    sys.exit(1)

# --- 2. Preprocessing ---
print("="*60)
print("STEP 2: Preprocessing (bandpass, whitening)...")

def gwpy_to_pycbc_timeseries(gwpy_ts):
    """Convert GWpy TimeSeries to PyCBC TimeSeries."""
    return PyCBC_TimeSeries(gwpy_ts.value, delta_t=gwpy_ts.dt.value, epoch=gwpy_ts.t0.value)

def preprocess_strain(ts, det, bandpass=(30, 250), whiten=True):
    print(f"Processing {det}: Bandpass {bandpass[0]}-{bandpass[1]} Hz, Whiten={whiten}")
    try:
        # Convert to PyCBC TimeSeries if needed
        if not isinstance(ts, PyCBC_TimeSeries):
            ts = gwpy_to_pycbc_timeseries(ts)
        # Bandpass filtering
        ts = highpass(ts, bandpass[0])
        ts = lowpass(ts, bandpass[1])
        # Optional whitening
        if whiten:
            seglen = 4  # seconds for PSD estimation
            print(f"  Estimating PSD for {det} (for whitening)...")
            psd = ts.psd(fftlength=seglen, method='median')
            psd = interpolate(psd, ts.delta_f)
            psd = inverse_spectrum_truncation(psd, int(2 * ts.sample_rate))
            print(f"  Whitening {det}...")
            ts = ts.whiten(psd=psd, avg_psd=False)
        print(f"  {det}: Output delta_t={ts.delta_t}, length={len(ts)}")
        return ts
    except Exception as e:
        print(f"  ERROR processing {det}: {e}")
        traceback.print_exc()
        return None

pycbc_strain = {}
for det in detectors:
    if strain_data.get(det) is not None:
        pycbc_strain[det] = preprocess_strain(strain_data[det], det, bandpass=bandpass, whiten=whiten)
    else:
        print(f"Skipping {det}: No data available.")
        pycbc_strain[det] = None

# --- 3. Template Generation ---
print("="*60)
print("STEP 3: Generating waveform templates...")

# Use H1 as reference for sample rate and length, fallback to L1 if needed
ref_det = 'H1' if pycbc_strain.get('H1') is not None else 'L1'
if pycbc_strain.get(ref_det) is None:
    print("ERROR: No valid reference detector for template generation. Exiting.")
    sys.exit(1)

data_sample_rate = sample_rates[ref_det]
data_num_samples = num_samples[ref_det]
data_delta_t = 1.0 / data_sample_rate

template_bank = {}

print(f"Generating templates for masses 20-30 Msun, zero spin, sample_rate={data_sample_rate} Hz, num_samples={data_num_samples}")

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates
        key = (m1, m2)
        print(f"  Generating template for m1={m1}, m2={m2}...", end='')
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=data_delta_t,
                                    f_lower=30.0,
                                    spin1z=0, spin2z=0)
            # Check if template is too short
            if len(hp) < 2:
                print(" FAILED (too short)")
                continue
            # Crop or pad to match data length
            if len(hp) > data_num_samples:
                hp = hp[:data_num_samples]
                print(f" cropped to {len(hp)} samples.", end='')
            elif len(hp) < data_num_samples:
                pad_len = data_num_samples - len(hp)
                hp = PyCBC_TimeSeries(np.pad(hp.numpy(), (pad_len, 0)), delta_t=hp.delta_t, epoch=hp.start_time - pad_len * hp.delta_t)
                print(f" padded to {len(hp)} samples.", end='')
            else:
                print(" length OK.", end='')
            if len(hp) != data_num_samples:
                print(" ERROR (length mismatch)")
                continue
            template_bank[key] = hp
            print(" Success.")
        except Exception as e:
            print(f" FAILED ({e})")
            traceback.print_exc()
            continue

if not template_bank:
    print("ERROR: No valid templates generated. Exiting.")
    sys.exit(1)

# --- 4. PSD Estimation and Matched Filtering ---
print("="*60)
print("STEP 4: PSD estimation and matched filtering...")

def adaptive_psd(strain, seg_len_sec=2.0, avg_method='median'):
    n_data = len(strain)
    sample_rate = 1.0 / strain.delta_t
    seglen = int(seg_len_sec * sample_rate)
    min_segments = 4  # Require at least 4 segments for a good PSD estimate

    # Adjust segment length if needed
    while n_data // seglen < min_segments and seglen > 32:
        seglen = seglen // 2
    if seglen < 32:
        seglen = max(32, n_data // 4)
    print(f"  PSD estimation: using segment length {seglen} samples ({seglen/sample_rate:.2f} s)")
    try:
        psd = welch(strain, seglen=seglen, avg_method=avg_method)
        return psd
    except Exception as e:
        print(f"    Welch PSD estimation failed: {e}")
        print("    Trying fallback segment length...")
        try:
            seglen = max(32, n_data // 4)
            psd = welch(strain, seglen=seglen, avg_method=avg_method)
            return psd
        except Exception as e2:
            print(f"    Fallback PSD estimation failed: {e2}")
            raise

results = {}

for det in detectors:
    print(f"\nProcessing detector {det}...")
    strain = pycbc_strain.get(det)
    if strain is None:
        print(f"  No strain data for {det}, skipping.")
        continue

    # Estimate PSD
    try:
        psd = adaptive_psd(strain)
    except Exception as e:
        print(f"  PSD estimation failed for {det}: {e}")
        traceback.print_exc()
        continue

    results[det] = {}

    # Matched filtering for each template
    for key, template in template_bank.items():
        m1, m2 = key
        print(f"    Matched filtering for template m1={m1}, m2={m2}...", end='')
        try:
            # Ensure template and data have same delta_t and length
            if template.delta_t != strain.delta_t:
                print(" delta_t mismatch, resampling...", end='')
                template = template.resample(strain.delta_t)
            if len(template) != len(strain):
                print(" length mismatch, cropping/padding...", end='')
                if len(template) > len(strain):
                    template = template[:len(strain)]
                else:
                    pad_len = len(strain) - len(template)
                    template = PyCBC_TimeSeries(np.pad(template.numpy(), (pad_len, 0)), delta_t=template.delta_t, epoch=template.start_time - pad_len * template.delta_t)
            # Matched filter
            snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=30)
            # Crop edge artifacts
            crop_samples = int(crop_edge_sec / strain.delta_t)
            if 2 * crop_samples >= len(snr):
                print(" SNR series too short after cropping, skipping.")
                continue
            snr_cropped = snr.crop(crop_samples, crop_samples)
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            results[det][key] = {'max_snr': float(max_snr), 'max_time': float(max_time)}
            print(f" max SNR={max_snr:.2f} at t={max_time:.3f}")
        except Exception as e:
            print(f" FAILED ({e})")
            traceback.print_exc()
            continue

print("="*60)
print("Analysis complete.")
print("Summary of maximum SNRs for each detector:")
for det in results:
    if not results[det]:
        print(f"  {det}: No successful matched filters.")
        continue
    best = max(results[det].items(), key=lambda x: x[1]['max_snr'])
    (m1, m2), res = best
    print(f"  {det}: Best template m1={m1}, m2={m2} -> max SNR={res['max_snr']:.2f} at t={res['max_time']:.3f}")

# Optionally, save results to file (uncomment if needed)
# import pickle
# with open("matched_filter_results.pkl", "wb") as f:
#     pickle.dump(results, f)