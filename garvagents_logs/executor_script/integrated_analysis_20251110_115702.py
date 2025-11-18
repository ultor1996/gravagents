# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter
import os
import sys

# --- Utility Functions ---
def save_npz(filename, **kwargs):
    try:
        np.savez_compressed(filename, **kwargs)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Parameters ---
t0 = 1126259462.4  # GW150914 GPS time
t_start = t0 - 8
t_end = t0 + 4
detectors = ['H1', 'L1']
f_low = 30
f_high = 250
mass_range = np.arange(20, 31, 1)
min_template_duration = 1.0  # seconds
crop_edge_sec = 0.2
output_dir = "gw150914_analysis_results"
ensure_dir(output_dir)

# --- Task 1: Data Loading & Preprocessing ---
print("\n=== Task 1: Data Loading & Preprocessing ===")
strain_data = {}
sample_rates = {}
num_samples = {}

for det in detectors:
    print(f"\nProcessing {det} data...")
    try:
        print(f"Fetching open data for {det} from {t_start} to {t_end}...")
        ts = TimeSeries.fetch_open_data(det, t_start, t_end, cache=True)
        print(f"Data fetched: {len(ts)} samples, duration: {ts.duration.value:.2f} s")

        print(f"Applying bandpass filter: {f_low}-{f_high} Hz...")
        ts_bp = ts.bandpass(f_low, f_high)

        print("Whitening the data...")
        ts_proc = ts_bp.whiten()

        delta_t = ts_proc.dt.value
        n_samples = len(ts_proc)
        print(f"Sample rate: {1/delta_t:.2f} Hz, Number of samples: {n_samples}")

        strain_data[det] = ts_proc
        sample_rates[det] = 1/delta_t
        num_samples[det] = n_samples

    except Exception as e:
        print(f"Error processing {det}: {e}")
        sys.exit(1)

# Save preprocessed data info
save_npz(os.path.join(output_dir, "strain_data_info.npz"),
         sample_rates=sample_rates, num_samples=num_samples)

# --- Task 2: Template Generation ---
print("\n=== Task 2: Template Generation ===")
ref_detector = 'H1'
sample_rate = sample_rates[ref_detector]
n_data_samples = num_samples[ref_detector]
templates = {}
template_params = []

print(f"Generating templates for mass pairs in 20–30 M_sun, zero spin...")
for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue
        params = {
            'mass1': m1,
            'mass2': m2,
            'spin1z': 0,
            'spin2z': 0,
            'delta_t': 1.0 / sample_rate,
            'f_lower': 30.0,
            'approximant': 'SEOBNRv4_opt'
        }
        try:
            hp, _ = get_td_waveform(**params)
            if len(hp) == 0:
                print(f"Skipping m1={m1}, m2={m2}: empty waveform.")
                continue

            min_samples = int(sample_rate * min_template_duration)
            if len(hp) < min_samples:
                print(f"Skipping m1={m1}, m2={m2}: waveform too short ({len(hp)/sample_rate:.2f} s).")
                continue

            # Crop to 1 s if longer
            if len(hp) > min_samples:
                hp = hp[-min_samples:]

            # Pad or truncate to match data length
            if len(hp) < n_data_samples:
                pad_width = n_data_samples - len(hp)
                hp = np.pad(hp, (pad_width, 0), 'constant')
            elif len(hp) > n_data_samples:
                hp = hp[-n_data_samples:]

            key = (m1, m2)
            templates[key] = hp
            template_params.append({'mass1': m1, 'mass2': m2})
            print(f"Template m1={m1}, m2={m2}: length={len(hp)}, stored.")

        except Exception as e:
            print(f"Error for m1={m1}, m2={m2}: {e}")

print(f"Total valid templates generated: {len(templates)}")
save_npz(os.path.join(output_dir, "templates.npz"),
         template_keys=np.array(list(templates.keys()), dtype=object),
         template_params=np.array(template_params, dtype=object),
         templates=np.array(list(templates.values()), dtype=object))

# --- Task 3: PSD Estimation ---
print("\n=== Task 3: PSD Estimation ===")
psds = {}

for det in strain_data:
    print(f"\nEstimating PSD for {det}...")
    data = strain_data[det].value
    n_data = num_samples[det]
    sample_rate = sample_rates[det]
    seg_len_sec = 2.0

    while True:
        seg_len_samples = int(seg_len_sec * sample_rate)
        if seg_len_samples > n_data:
            seg_len_sec /= 2
            print(f"Segment too long for {det}, halving to {seg_len_sec:.2f} s...")
            if seg_len_sec < 1.0 / sample_rate * 32:
                seg_len_samples = n_data // 4
                print(f"Fallback: using segment length {seg_len_samples} samples for {det}.")
                break
        elif seg_len_samples < 32:
            seg_len_sec /= 2
            print(f"Segment <32 samples for {det}, halving to {seg_len_sec:.2f} s...")
            if seg_len_sec < 1.0 / sample_rate * 32:
                seg_len_samples = n_data // 4
                print(f"Fallback: using segment length {seg_len_samples} samples for {det}.")
                break
        else:
            break

    try:
        print(f"Using segment length {seg_len_samples} samples ({seg_len_samples/sample_rate:.2f} s) for {det}.")
        psd = welch(
            data,
            seg_len=seg_len_samples,
            sample_rate=sample_rate,
            avg_method='median'
        )
        psds[det] = psd
        print(f"PSD estimation complete for {det}. PSD length: {len(psd)}")
    except Exception as e:
        print(f"Error estimating PSD for {det}: {e}")
        sys.exit(1)

# Save PSDs
for det in psds:
    try:
        np.save(os.path.join(output_dir, f"psd_{det}.npy"), psds[det].numpy())
        print(f"PSD for {det} saved.")
    except Exception as e:
        print(f"Error saving PSD for {det}: {e}")

# --- Task 4: Matched Filtering ---
print("\n=== Task 4: Matched Filtering ===")
pycbc_strain = {}
for det in strain_data:
    print(f"Converting {det} strain to PyCBC TimeSeries...")
    try:
        arr = strain_data[det].value
        delta_t = strain_data[det].dt.value
        epoch = strain_data[det].t0.value
        pycbc_strain[det] = PyCBC_TimeSeries(arr, delta_t=delta_t, epoch=epoch)
        print(f"{det} conversion successful.")
    except Exception as e:
        print(f"Error converting {det}: {e}")
        sys.exit(1)

pycbc_templates = {}
for key, arr in templates.items():
    try:
        delta_t = pycbc_strain['H1'].delta_t
        epoch = pycbc_strain['H1'].start_time
        pycbc_templates[key] = PyCBC_TimeSeries(arr, delta_t=delta_t, epoch=epoch)
    except Exception as e:
        print(f"Error converting template {key}: {e}")

snr_results = {}  # {(det, (m1, m2)): {'max_abs_snr': ..., 'time': ...}}

for det in pycbc_strain:
    print(f"\nMatched filtering for {det}...")
    data = pycbc_strain[det]
    psd = psds[det]
    for key, template in pycbc_templates.items():
        try:
            snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=30.0)
            crop_samples = int(crop_edge_sec / data.delta_t)
            if 2 * crop_samples >= len(snr):
                print(f"Skipping template {key} for {det}: not enough samples after cropping.")
                continue
            snr_cropped = snr.crop(crop_samples, crop_samples)
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_abs_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            snr_results[(det, key)] = {'max_abs_snr': float(max_abs_snr), 'time': float(max_time)}
            print(f"{det} template {key}: max|SNR|={max_abs_snr:.2f} at t={max_time}")
        except Exception as e:
            print(f"Error matched filtering {det} template {key}: {e}")

# Save SNR results
try:
    import json
    snr_results_json = {f"{det}_{key[0]}_{key[1]}": val for (det, key), val in snr_results.items()}
    with open(os.path.join(output_dir, "snr_results.json"), "w") as f:
        json.dump(snr_results_json, f, indent=2)
    print(f"SNR results saved to {os.path.join(output_dir, 'snr_results.json')}")
except Exception as e:
    print(f"Error saving SNR results: {e}")

print("\n=== Analysis Complete ===")
print(f"All results saved in directory: {output_dir}")