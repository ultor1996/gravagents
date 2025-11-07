# === Imports ===
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter
import traceback

# === Section 1: Download and Filter GW150914 Data ===
print("\n=== 1. Downloading and Filtering GW150914 Data ===")
GW150914_GPS = 1126259462.4
T_START = GW150914_GPS - 8
T_END = GW150914_GPS + 4
DETECTORS = ['H1', 'L1']
BANDPASS_LOW = 30
BANDPASS_HIGH = 250

strain_data = {}
sample_rates = {}
n_data = {}

for det in DETECTORS:
    print(f"\nDownloading data for {det} from {T_START} to {T_END}...")
    try:
        ts = TimeSeries.get(f'{det}:GWOSC-4KHZ_R1_STRAIN', T_START, T_END, cache=True)
        print(f"Downloaded {len(ts)} samples for {det}.")
        print(f"Applying {BANDPASS_LOW}-{BANDPASS_HIGH} Hz bandpass filter to {det} data...")
        ts_bp = ts.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print(f"Bandpass filter applied for {det}.")
        delta_t = ts_bp.dt.value
        n_samples = len(ts_bp)
        print(f"{det} sample rate: {1/delta_t:.1f} Hz, n_data: {n_samples}")
        strain_data[det] = ts_bp
        sample_rates[det] = 1/delta_t
        n_data[det] = n_samples
    except Exception as e:
        print(f"Error processing {det}: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to process {det} data.")

# Consistency check for n_data
if n_data['H1'] != n_data['L1']:
    raise ValueError(f"n_data mismatch: H1={n_data['H1']}, L1={n_data['L1']}")

# === Section 2: Convert GWpy TimeSeries to PyCBC TimeSeries ===
print("\n=== 2. Converting GWpy TimeSeries to PyCBC TimeSeries ===")
def check_and_convert_gwpy_to_pycbc(strain_data):
    detectors = ['H1', 'L1']
    pycbc_data = {}
    delta_ts = {det: strain_data[det].dt.value for det in detectors}
    epochs = {det: strain_data[det].t0.value for det in detectors}
    print("Checking delta_t and epoch consistency between detectors...")
    if not (delta_ts['H1'] == delta_ts['L1']):
        raise ValueError(f"Sample rates differ: H1 delta_t={delta_ts['H1']}, L1 delta_t={delta_ts['L1']}")
    if not (epochs['H1'] == epochs['L1']):
        raise ValueError(f"Epochs differ: H1 epoch={epochs['H1']}, L1 epoch={epochs['L1']}")
    print("delta_t and epoch are consistent. Proceeding with conversion...")
    for det in detectors:
        try:
            print(f"Converting {det} GWpy TimeSeries to PyCBC TimeSeries...")
            arr = strain_data[det].value
            delta_t = delta_ts[det]
            epoch = epochs[det]
            pycbc_ts = PyCBC_TimeSeries(arr, delta_t=delta_t, epoch=epoch)
            pycbc_data[det] = pycbc_ts
            print(f"{det} conversion successful. PyCBC TimeSeries length: {len(pycbc_ts)}")
        except Exception as e:
            print(f"Error converting {det}: {e}")
            traceback.print_exc()
            raise
    return pycbc_data

try:
    pycbc_data = check_and_convert_gwpy_to_pycbc(strain_data)
except Exception as e:
    print(f"Failed to convert GWpy to PyCBC TimeSeries: {e}")
    raise

# Use these for later steps
delta_t = pycbc_data['H1'].delta_t
n_data_val = len(pycbc_data['H1'])

# === Section 3: Generate PyCBC Waveform Templates ===
print("\n=== 3. Generating PyCBC Waveform Templates ===")
mass_range = range(20, 31)
templates = {}
n_templates_attempted = 0
n_templates_success = 0

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only unique (m1 >= m2) pairs
        key = (m1, m2)
        n_templates_attempted += 1
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=delta_t,
                                    f_lower=30,
                                    spin1z=0, spin2z=0)
            max_samples = int(1.0 / delta_t)
            if len(hp) > max_samples:
                print(f"Cropping template {key} from {len(hp)} to {max_samples} samples (1 s)...")
                hp = hp[:max_samples]
            # Pad or truncate to n_data
            if len(hp) < n_data_val:
                pad_width = n_data_val - len(hp)
                hp_padded = np.pad(hp.numpy(), (pad_width, 0), 'constant')
                hp_final = hp.copy()
                hp_final.data = hp_padded
                print(f"Padded template {key} to {n_data_val} samples.")
            elif len(hp) > n_data_val:
                hp_final = hp[-n_data_val:]
                print(f"Truncated template {key} to {n_data_val} samples.")
            else:
                hp_final = hp
                print(f"Template {key} already has {n_data_val} samples.")
            templates[key] = hp_final
            n_templates_success += 1
        except Exception as e:
            print(f"Error generating template for (m1={m1}, m2={m2}): {e}")
            traceback.print_exc()
print(f"Generated {n_templates_success} templates out of {n_templates_attempted} attempted.")

if len(templates) == 0:
    raise RuntimeError("No templates generated. Aborting.")

# === Section 4: Estimate PSD for Each Detector ===
print("\n=== 4. Estimating PSD for Each Detector ===")
psds = {}
seg_len_sec = 2

for det in DETECTORS:
    print(f"\nEstimating PSD for {det}...")
    data = pycbc_data[det]
    n = len(data)
    sample_rate = 1.0 / data.delta_t
    seg_len = int(seg_len_sec * sample_rate)
    print(f"Initial seg_len for {det}: {seg_len} samples ({seg_len/sample_rate:.2f} s)")
    while seg_len > n:
        seg_len = seg_len // 2
        print(f"  seg_len too long, halved to {seg_len} samples")
    if seg_len < 32:
        seg_len = max(32, n // 4)
        print(f"  seg_len too short, set to {seg_len} samples")
    if seg_len > n:
        seg_len = n // 4
        print(f"  seg_len fallback to {seg_len} samples")
    try:
        print(f"Using seg_len={seg_len} samples for {det} PSD estimation...")
        psd = welch(data, seg_len=seg_len, avg_method='median')
        psds[det] = psd
        print(f"PSD estimation for {det} complete. PSD length: {len(psd)}")
    except Exception as e:
        print(f"Error estimating PSD for {det}: {e}")
        traceback.print_exc()
        raise

# === Section 5: Matched Filtering and SNR Extraction ===
print("\n=== 5. Matched Filtering and SNR Extraction ===")
snr_results = {}
crop_sec = 0.2

for det in DETECTORS:
    data = pycbc_data[det]
    psd = psds[det]
    delta_t = data.delta_t
    crop_samples = int(crop_sec / delta_t)
    n_data_val = len(data)
    print(f"\nMatched filtering for detector {det} (crop {crop_samples} samples at each edge)...")
    for key, template in templates.items():
        try:
            snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=30)
            if 2 * crop_samples >= len(snr):
                print(f"Warning: SNR series for {det}, template {key} too short to crop. Skipping.")
                continue
            snr_cropped = snr[crop_samples:-crop_samples]
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_abs_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            snr_results[(det, key)] = {
                'max_abs_snr': float(max_abs_snr),
                'time': float(max_time),
                'snr_series': snr_cropped
            }
            print(f"{det} {key}: max|SNR|={max_abs_snr:.2f} at t={max_time:.4f}")
        except Exception as e:
            print(f"Error in matched filtering for {det}, template {key}: {e}")
            traceback.print_exc()

print("\n=== Workflow Complete ===")
print(f"Total SNR results computed: {len(snr_results)}")

# === Optional: Save Results ===
# Uncomment and modify the following lines to save results as needed.
# np.save('snr_results.npy', snr_results)
# np.save('templates.npy', templates)
# np.save('psds.npy', psds)