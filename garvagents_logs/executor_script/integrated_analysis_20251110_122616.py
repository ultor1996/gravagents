# --- Imports ---
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.psd import welch
from pycbc.filter import matched_filter

# --- Constants and Configuration ---
GW150914_GPS = 1126259462.4
DURATION = 12  # seconds
BANDPASS_LOW = 20
BANDPASS_HIGH = 250
WHITEN = True  # Set to False if whitening is not desired
MASS_RANGE = np.arange(10, 31, 1)  # 10, 11, ..., 30
OUTPUT_H5 = "matched_filter_results.h5"
EDGE_CROP_SEC = 0.2  # seconds to crop from each edge in SNR

# --- Task 1: Fetch and Preprocess Data ---
print("="*60)
print("TASK 1: Fetching and preprocessing strain data")
print("="*60)

detectors = ['H1', 'L1']
strain_data = {}
sample_rates = {}
num_samples = {}

for det in detectors:
    print(f"\nFetching data for {det}...")
    try:
        start = GW150914_GPS - DURATION / 2
        end = GW150914_GPS + DURATION / 2
        ts = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"  Data fetched for {det}: {len(ts)} samples, sample rate {ts.sample_rate.value} Hz")
    except Exception as e:
        print(f"  ERROR fetching data for {det}: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        num_samples[det] = None
        continue

    try:
        ts_bp = ts.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print(f"  Bandpass filter applied ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)")
    except Exception as e:
        print(f"  ERROR applying bandpass filter for {det}: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        num_samples[det] = None
        continue

    try:
        if WHITEN:
            ts_proc = ts_bp.whiten()
            print("  Whitening applied")
        else:
            ts_proc = ts_bp
            print("  Whitening skipped")
    except Exception as e:
        print(f"  ERROR whitening data for {det}: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        num_samples[det] = None
        continue

    try:
        strain_np = ts_proc.value
        if not isinstance(strain_np, np.ndarray):
            strain_np = np.array(strain_np)
        strain_data[det] = strain_np
        sample_rates[det] = ts_proc.sample_rate.value
        num_samples[det] = len(strain_np)
        print(f"  Data for {det}: {num_samples[det]} samples, sample rate {sample_rates[det]} Hz")
    except Exception as e:
        print(f"  ERROR converting data for {det} to numpy array: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        num_samples[det] = None

print("\nSummary of fetched and processed data:")
for det in detectors:
    print(f"  {det}: {num_samples[det]} samples, sample rate {sample_rates[det]} Hz")

# --- Task 2: Generate Templates ---
print("\n" + "="*60)
print("TASK 2: Generating waveform templates")
print("="*60)

# Use H1's data properties for template matching (fallback to L1 if H1 missing)
if strain_data['H1'] is not None:
    det_for_templates = 'H1'
elif strain_data['L1'] is not None:
    det_for_templates = 'L1'
else:
    raise RuntimeError("No valid strain data available for template generation.")

data_length = num_samples[det_for_templates]
delta_t = 1.0 / sample_rates[det_for_templates]

templates = []
template_params = []

print(f"\nGenerating templates for m1, m2 in 10–30 M⊙ (m1 >= m2), zero spins, SEOBNRv4_opt...")
for m1 in MASS_RANGE:
    for m2 in MASS_RANGE:
        if m1 < m2:
            continue  # Only m1 >= m2
        params = {
            'mass1': float(m1),
            'mass2': float(m2),
            'spin1z': 0.0,
            'spin2z': 0.0,
            'approximant': 'SEOBNRv4_opt',
            'f_lower': 10.0,
            'delta_t': delta_t
        }
        try:
            hp, _ = get_td_waveform(**params)
        except Exception as e:
            print(f"  Skipping m1={m1}, m2={m2}: waveform generation failed ({e})")
            continue

        duration = hp.duration
        if duration < 0.2:
            print(f"  Skipping m1={m1}, m2={m2}: duration {duration:.3f}s < 0.2s")
            continue

        hp_np = hp.numpy()
        if len(hp_np) < data_length:
            pad_width = data_length - len(hp_np)
            hp_np = np.pad(hp_np, (pad_width, 0), 'constant')
        elif len(hp_np) > data_length:
            hp_np = hp_np[-data_length:]

        try:
            template_ts = PyCBC_TimeSeries(hp_np, delta_t=delta_t, epoch=hp.start_time)
        except Exception as e:
            print(f"  Skipping m1={m1}, m2={m2}: TimeSeries conversion failed ({e})")
            continue

        templates.append(template_ts)
        template_params.append({'m1': m1, 'm2': m2, 'duration': duration})
        print(f"  Template m1={m1}, m2={m2}: duration={duration:.3f}s, samples={len(hp_np)}")

print(f"\nGenerated {len(templates)} valid templates.")

if len(templates) == 0:
    raise RuntimeError("No valid templates generated. Exiting.")

# --- Task 3: Estimate PSDs ---
print("\n" + "="*60)
print("TASK 3: Estimating PSDs")
print("="*60)

def next_lower_power_of_2(n):
    """Return the largest power of 2 less than or equal to n."""
    return 2 ** int(np.floor(np.log2(n)))

psds = {}

for det in detectors:
    strain = strain_data[det]
    sr = sample_rates[det]
    n_data = num_samples[det]
    print(f"\nEstimating PSD for {det}: {n_data} samples, sample rate {sr} Hz")

    if strain is None or sr is None or n_data is None:
        print(f"  Skipping PSD estimation for {det}: missing data.")
        psds[det] = None
        continue

    max_seg_len = n_data // 2
    seg_len = next_lower_power_of_2(max_seg_len)
    if seg_len < 32:
        seg_len = 32
    print(f"  Using segment length: {seg_len} samples")

    try:
        if not isinstance(strain, PyCBC_TimeSeries):
            strain_ts = PyCBC_TimeSeries(strain, delta_t=1.0/sr)
        else:
            strain_ts = strain

        psd = welch(strain_ts, seg_len=seg_len, avg_method='median')
        print(f"  PSD estimated for {det} (len={len(psd)}, delta_f={psd.delta_f})")
        psds[det] = psd
    except Exception as e:
        print(f"  PSD estimation failed for {det} with seg_len={seg_len}: {e}")
        seg_len2 = next_lower_power_of_2(n_data // 4)
        if seg_len2 < 32:
            seg_len2 = 32
        print(f"  Retrying with segment length: {seg_len2} samples")
        try:
            psd = welch(strain_ts, seg_len=seg_len2, avg_method='median')
            print(f"  PSD estimated for {det} (len={len(psd)}, delta_f={psd.delta_f})")
            psds[det] = psd
        except Exception as e2:
            print(f"  PSD estimation failed again for {det}: {e2}")
            psds[det] = None

print("\nPSD estimation summary:")
for det in detectors:
    if psds[det] is not None:
        print(f"  {det}: PSD length={len(psds[det])}, delta_f={psds[det].delta_f}")
    else:
        print(f"  {det}: PSD estimation FAILED")

# --- Task 4: Matched Filtering and Saving ---
print("\n" + "="*60)
print("TASK 4: Matched filtering and saving results")
print("="*60)

summary_dtype = np.dtype([
    ('m1', 'f4'), ('m2', 'f4'),
    ('max_snr_H1', 'f4'), ('t_H1', 'f8'),
    ('max_snr_L1', 'f4'), ('t_L1', 'f8')
])
summary_table = []
template_arrs = []
psd_arrs = {}

print("\nStarting matched filtering for all templates and detectors...")

for i, (template, params) in enumerate(zip(templates, template_params)):
    m1, m2 = params['m1'], params['m2']
    row = [m1, m2, np.nan, np.nan, np.nan, np.nan]
    print(f"\nTemplate {i+1}/{len(templates)}: m1={m1}, m2={m2}")

    template_arrs.append(template.numpy())

    for det in detectors:
        strain = strain_data.get(det)
        sr = sample_rates.get(det)
        psd = psds.get(det)
        if strain is None or sr is None or psd is None:
            print(f"  Skipping {det}: missing strain, sample rate, or PSD.")
            continue

        try:
            if not isinstance(strain, PyCBC_TimeSeries):
                strain_ts = PyCBC_TimeSeries(strain, delta_t=1.0/sr)
            else:
                strain_ts = strain

            snr = matched_filter(template, strain_ts, psd=psd, low_frequency_cutoff=20.0)
            if len(snr) == 0:
                print(f"  {det}: Empty SNR series, skipping.")
                continue

            crop_samples = int(EDGE_CROP_SEC * sr)
            if len(snr) > 2 * crop_samples:
                snr_cropped = snr.crop(crop_samples, crop_samples)
                print(f"  {det}: Cropped {crop_samples} samples from each edge.")
            else:
                snr_cropped = snr
                print(f"  {det}: Not enough samples to crop edges.")

            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            print(f"  {det}: max|SNR|={max_snr:.2f} at t={max_time:.4f}")

            if det == 'H1':
                row[2] = max_snr
                row[3] = max_time
            elif det == 'L1':
                row[4] = max_snr
                row[5] = max_time

        except Exception as e:
            print(f"  {det}: Matched filtering failed: {e}")
            continue

    summary_table.append(tuple(row))

for det in detectors:
    psd = psds.get(det)
    if psd is not None:
        psd_arrs[det] = psd.numpy()
    else:
        psd_arrs[det] = None

print(f"\nSaving results to {OUTPUT_H5} ...")
try:
    with h5py.File(OUTPUT_H5, 'w') as f:
        f.create_dataset('templates', data=np.stack(template_arrs))
        f.create_dataset('template_params', data=np.array(
            [(p['m1'], p['m2'], p['duration']) for p in template_params],
            dtype=[('m1', 'f4'), ('m2', 'f4'), ('duration', 'f4')]
        ))
        for det in detectors:
            if psd_arrs[det] is not None:
                f.create_dataset(f'psd_{det}', data=psd_arrs[det])
        f.create_dataset('summary', data=np.array(summary_table, dtype=summary_dtype))
    print("Results saved successfully.")
except Exception as e:
    print(f"ERROR saving to HDF5: {e}")

print("\nAll tasks completed.")