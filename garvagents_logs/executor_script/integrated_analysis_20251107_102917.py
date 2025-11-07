# --- Imports ---
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter
import numpy as np

# --- PARAMETERS ---
gps_start = 1126259462  # Example GPS start time (GW150914)
duration = 4            # Duration in seconds
detectors = ['H1', 'L1']
bandpass_low = 30
bandpass_high = 250
mass_range = range(20, 31)  # 20 to 30 inclusive

# --- SECTION 1: Data Loading and Filtering ---
print("\n=== 1. Loading and Filtering Strain Data ===")
gwpy_data = {}
pycbc_data = {}
sample_rates = {}
n_data = {}

try:
    print("Loading raw strain data for H1 and L1 using GWpy...")
    for det in detectors:
        print(f"  Loading {det}...")
        ts = TimeSeries.fetch_open_data(det, gps_start, gps_start + duration, cache=False)
        print(f"    Loaded {len(ts)} samples for {det}.")
        gwpy_data[det] = ts
except Exception as e:
    print(f"Error loading data: {e}")
    raise

try:
    print("Applying 30–250 Hz bandpass filter (Butterworth, order=4)...")
    for det in detectors:
        ts = gwpy_data[det]
        filtered = ts.bandpass(bandpass_low, bandpass_high, filtfilt=True, order=4)
        gwpy_data[det] = filtered
        print(f"    Filtered {det}.")
except Exception as e:
    print(f"Error during filtering: {e}")
    raise

try:
    print("Recording sample rate and number of data points...")
    for det in detectors:
        ts = gwpy_data[det]
        sample_rates[det] = ts.sample_rate.value
        n_data[det] = len(ts)
        print(f"    {det}: sample_rate = {sample_rates[det]} Hz, n_data = {n_data[det]}")
except Exception as e:
    print(f"Error recording sample rate or n_data: {e}")
    raise

try:
    print("Ensuring both detectors have matching delta_t and epoch...")
    delta_t = 1.0 / min(sample_rates.values())
    epoch = max(ts.t0.value for ts in gwpy_data.values())
    print(f"  Using delta_t = {delta_t}, epoch = {epoch}")

    for det in detectors:
        ts = gwpy_data[det]
        # Resample if needed
        if ts.sample_rate.value != 1.0 / delta_t:
            print(f"    Resampling {det} to {1.0/delta_t} Hz...")
            ts = ts.resample(1.0/delta_t)
        # Trim or pad to align epochs
        if ts.t0.value != epoch:
            print(f"    Shifting {det} to epoch {epoch}...")
            ts = ts.crop(epoch, epoch + duration)
            if len(ts) < int(duration / delta_t):
                pad_len = int(duration / delta_t) - len(ts)
                ts = ts.append(np.zeros(pad_len), times=ts.times.value[-1] + delta_t)
        gwpy_data[det] = ts

    lengths = [len(ts) for ts in gwpy_data.values()]
    epochs = [ts.t0.value for ts in gwpy_data.values()]
    assert all(l == lengths[0] for l in lengths), "Data lengths do not match!"
    assert all(e == epochs[0] for e in epochs), "Epochs do not match!"

except Exception as e:
    print(f"Error aligning data: {e}")
    raise

try:
    print("Converting to PyCBC TimeSeries objects...")
    for det in detectors:
        ts = gwpy_data[det]
        pycbc_ts = PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)
        pycbc_data[det] = pycbc_ts
        print(f"    Converted {det} to PyCBC TimeSeries.")
except Exception as e:
    print(f"Error converting to PyCBC TimeSeries: {e}")
    raise

# Save results for downstream tasks
H1_strain = pycbc_data['H1']
L1_strain = pycbc_data['L1']
H1_sample_rate = sample_rates['H1']
L1_sample_rate = sample_rates['L1']
H1_n_data = n_data['H1']
L1_n_data = n_data['L1']

print("Data loading and filtering complete.")
print(f"H1: sample_rate={H1_sample_rate}, n_data={H1_n_data}")
print(f"L1: sample_rate={L1_sample_rate}, n_data={L1_n_data}")

# --- SECTION 2: Template Generation and Alignment ---
print("\n=== 2. Generating and Aligning Waveform Templates ===")
# Use H1's parameters as reference (should be same as L1)
delta_t = H1_strain.delta_t
n_data = len(H1_strain)

templates = {}
for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only unique pairs with m1 >= m2
        key = (m1, m2)
        try:
            print(f"  Generating template for m1={m1} M☉, m2={m2} M☉...")
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=delta_t,
                                    f_lower=20.0,
                                    spin1z=0, spin2z=0)
            # Crop to 1s if longer
            if hp.duration > 1.0:
                print(f"    Cropping template to 1s (original duration: {hp.duration:.3f}s)...")
                hp = hp.crop(hp.start_time, hp.start_time + 1.0)
            # Pad or truncate to match n_data
            if len(hp) < n_data:
                pad_len = n_data - len(hp)
                print(f"    Padding template with {pad_len} zeros...")
                hp = PyCBC_TimeSeries(np.pad(hp.numpy(), (0, pad_len)), delta_t=delta_t, epoch=hp.start_time)
            elif len(hp) > n_data:
                print(f"    Truncating template from {len(hp)} to {n_data} samples...")
                hp = PyCBC_TimeSeries(hp.numpy()[:n_data], delta_t=delta_t, epoch=hp.start_time)
            assert len(hp) == n_data, f"Template length mismatch: {len(hp)} != {n_data}"
            templates[key] = hp
        except Exception as e:
            print(f"    Error generating template for (m1={m1}, m2={m2}): {e}")

print(f"Generated {len(templates)} templates.")
waveform_templates = templates

# --- SECTION 3: PSD Estimation ---
print("\n=== 3. Estimating Power Spectral Densities (PSDs) ===")
psds = {}
strains = {'H1': H1_strain, 'L1': L1_strain}
sample_rates = {'H1': H1_sample_rate, 'L1': L1_sample_rate}
n_datas = {'H1': H1_n_data, 'L1': L1_n_data}

for det in detectors:
    print(f"Estimating PSD for {det}...")
    strain = strains[det]
    sample_rate = sample_rates[det]
    n_data = n_datas[det]
    seg_len_sec = 2.0
    min_samples = 32
    found_valid = False

    while seg_len_sec >= strain.delta_t:
        seg_len = int(seg_len_sec * sample_rate)
        if seg_len > n_data:
            print(f"  seg_len ({seg_len}) > n_data ({n_data}), halving seg_len_sec...")
            seg_len_sec /= 2
            continue
        if seg_len < min_samples:
            print(f"  seg_len ({seg_len}) < {min_samples} samples, halving seg_len_sec...")
            seg_len_sec /= 2
            continue
        found_valid = True
        break

    if not found_valid:
        seg_len = n_data // 4
        print(f"  No valid seg_len found, using fallback seg_len = {seg_len}")

    try:
        print(f"  Using seg_len = {seg_len} samples ({seg_len/sample_rate:.3f} s)")
        psd = welch(strain, seg_len=seg_len, avg_method='median')
        psds[det] = psd
        print(f"  PSD estimation for {det} complete. PSD length: {len(psd)}")
    except Exception as e:
        print(f"  Error estimating PSD for {det}: {e}")
        psds[det] = None

H1_psd = psds['H1']
L1_psd = psds['L1']

# --- SECTION 4: Matched Filtering and SNR Extraction ---
print("\n=== 4. Matched Filtering and SNR Extraction ===")
psds = {'H1': H1_psd, 'L1': L1_psd}
matched_filter_results = {det: {} for det in detectors}
crop_sec = 0.2

for det in detectors:
    print(f"\nPerforming matched filtering for {det}...")
    strain = strains[det]
    psd = psds[det]
    for key, template in waveform_templates.items():
        m1, m2 = key
        try:
            snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=30.0)
            crop_samples = int(crop_sec / strain.delta_t)
            if 2 * crop_samples >= len(snr):
                print(f"  Warning: Not enough samples to crop for template (m1={m1}, m2={m2}) in {det}. Skipping.")
                continue
            snr_cropped = snr.crop(crop_sec, crop_sec)
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_abs_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            matched_filter_results[det][key] = {
                'max_abs_snr': float(max_abs_snr),
                'time': float(max_time)
            }
            print(f"  (m1={m1}, m2={m2}): max|SNR|={max_abs_snr:.2f} at t={max_time:.4f}")
        except Exception as e:
            print(f"  Error for template (m1={m1}, m2={m2}) in {det}: {e}")

H1_matched_filter_results = matched_filter_results['H1']
L1_matched_filter_results = matched_filter_results['L1']

print("\n=== Pipeline Complete ===")
print(f"H1: {len(H1_matched_filter_results)} matched filter results")
print(f"L1: {len(L1_matched_filter_results)} matched filter results")

# Optionally, save results to disk (uncomment if needed)
# import pickle
# with open("matched_filter_results.pkl", "wb") as f:
#     pickle.dump(matched_filter_results, f)