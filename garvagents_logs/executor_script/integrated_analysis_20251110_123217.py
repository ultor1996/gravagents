# --- Imports ---
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter

# --- 1. Data Loading ---
print("\n=== 1. Fetching GW150914 Strain Data ===")
MERGER_GPS = 1126259462.4
WINDOW_BEFORE = 8
WINDOW_AFTER = 4
start_time = MERGER_GPS - WINDOW_BEFORE
end_time = MERGER_GPS + WINDOW_AFTER
detectors = ['H1', 'L1']
strain_data = {}
sample_rates = {}
num_samples = {}

for det in detectors:
    print(f"Fetching data for {det} from {start_time} to {end_time}...")
    try:
        ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        strain_data[det] = ts
        sample_rates[det] = ts.sample_rate.value
        num_samples[det] = len(ts)
        print(f"  {det}: Sample rate = {sample_rates[det]} Hz, Number of samples = {num_samples[det]}")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        strain_data[det] = None
        sample_rates[det] = None
        num_samples[det] = None

if any(strain_data[det] is None for det in detectors):
    raise RuntimeError("Failed to fetch data for one or more detectors. Exiting.")

# --- 2. Preprocessing ---
print("\n=== 2. Preprocessing: Bandpass Filtering and Whitening ===")
bandpass_low = 20
bandpass_high = 250
whiten_data = False  # Set to True to enable whitening

# Check sample rates and lengths
h1_sr = sample_rates['H1']
l1_sr = sample_rates['L1']
h1_len = num_samples['H1']
l1_len = num_samples['L1']

print("Checking sample rates and lengths...")
if h1_sr != l1_sr:
    print(f"Sample rates differ: H1={h1_sr}, L1={l1_sr}. Resampling to match.")
    target_sr = min(h1_sr, l1_sr)
    for det in detectors:
        try:
            strain_data[det] = strain_data[det].resample(target_sr)
            sample_rates[det] = strain_data[det].sample_rate.value
            num_samples[det] = len(strain_data[det])
            print(f"  {det} resampled to {target_sr} Hz.")
        except Exception as e:
            print(f"Error resampling {det}: {e}")
else:
    target_sr = h1_sr
    print("Sample rates are consistent.")

# Now check lengths
h1_len = num_samples['H1']
l1_len = num_samples['L1']
if h1_len != l1_len:
    print(f"Lengths differ: H1={h1_len}, L1={l1_len}. Trimming to match.")
    min_len = min(h1_len, l1_len)
    for det in detectors:
        try:
            strain_data[det] = strain_data[det][:min_len]
            num_samples[det] = len(strain_data[det])
            print(f"  {det} trimmed to {min_len} samples.")
        except Exception as e:
            print(f"Error trimming {det}: {e}")
else:
    print("Lengths are consistent.")

# Bandpass filter using GWpy
filtered_data = {}
for det in detectors:
    print(f"Applying {bandpass_low}-{bandpass_high} Hz bandpass filter to {det}...")
    try:
        filtered = strain_data[det].bandpass(bandpass_low, bandpass_high)
        filtered_data[det] = filtered
        print(f"  {det} bandpass filtering complete.")
    except Exception as e:
        print(f"Error bandpass filtering {det}: {e}")
        filtered_data[det] = None

if any(filtered_data[det] is None for det in detectors):
    raise RuntimeError("Bandpass filtering failed for one or more detectors. Exiting.")

# Optional whitening (not used for main analysis)
if whiten_data:
    whitened_data = {}
    for det in detectors:
        print(f"Whitening {det} data using PyCBC...")
        try:
            arr = filtered_data[det].value
            delta_t = 1.0 / filtered_data[det].sample_rate.value
            pycbc_ts = PyCBC_TimeSeries(arr, delta_t=delta_t)
            psd = pycbc_ts.psd(4 * filtered_data[det].sample_rate.value)
            psd = psd.interpolate(len(pycbc_ts) // 2 + 1)
            psd = psd.trim(20, 250)
            whitened = pycbc_ts.whiten(4, 2, psd=psd)
            whitened_data[det] = whitened
            print(f"  {det} whitening complete.")
        except Exception as e:
            print(f"Error whitening {det}: {e}")
            whitened_data[det] = None
else:
    whitened_data = None

# --- 3. Template Generation ---
print("\n=== 3. Generating Waveform Templates ===")
reference_detector = 'H1'
data_length = num_samples[reference_detector]
delta_t = 1.0 / sample_rates[reference_detector]
mass_range = np.arange(10, 31)
templates = {}
template_params = []

for m1 in mass_range:
    for m2 in mass_range:
        if m1 < m2:
            continue
        params = {
            'mass1': m1,
            'mass2': m2,
            'spin1z': 0,
            'spin2z': 0,
            'delta_t': delta_t,
            'f_lower': 10,
            'approximant': 'SEOBNRv4_opt'
        }
        try:
            hp, hc = get_td_waveform(**params)
            duration = len(hp) * hp.delta_t
            if duration < 0.2:
                print(f"  Skipping (m1={m1}, m2={m2}): duration {duration:.3f} s < 0.2 s")
                continue
            if len(hp) < data_length:
                pad_width = data_length - len(hp)
                hp_padded = np.pad(hp.numpy(), (0, pad_width), 'constant')
            else:
                hp_padded = hp.numpy()[:data_length]
            templates[(m1, m2)] = hp_padded
            template_params.append({'m1': m1, 'm2': m2, 'duration': duration})
            print(f"  Generated template for (m1={m1}, m2={m2}), duration {duration:.3f} s")
        except Exception as e:
            print(f"  Error generating template for (m1={m1}, m2={m2}): {e}")

print(f"Total templates generated: {len(templates)}")
if len(templates) == 0:
    raise RuntimeError("No templates generated. Exiting.")

# --- 4. PSD Estimation ---
print("\n=== 4. Estimating PSDs ===")
psds = {}
psd_params = {}

for det in detectors:
    print(f"Estimating PSD for {det}...")
    try:
        ts = filtered_data[det]
        arr = ts.value
        delta_t = 1.0 / ts.sample_rate.value
        pycbc_ts = PyCBC_TimeSeries(arr, delta_t=delta_t)
        n_samples = len(pycbc_ts)
        max_seg_len = n_samples // 2
        seg_len = 2 ** int(np.floor(np.log2(max_seg_len)))
        seg_len = max(seg_len, 32)
        print(f"  Data length: {n_samples}, segment length: {seg_len}")
        psd = welch(pycbc_ts, seg_len=seg_len, avg_method='median')
        psds[det] = psd
        psd_params[det] = {
            'seg_len': seg_len,
            'delta_f': psd.delta_f,
            'length': len(psd)
        }
        print(f"  PSD estimated: length={len(psd)}, delta_f={psd.delta_f:.6f} Hz")
    except Exception as e:
        print(f"  Error estimating PSD for {det}: {e}")
        psds[det] = None
        psd_params[det] = None

if any(psds[det] is None for det in detectors):
    print("Warning: PSD estimation failed for one or more detectors. Those detectors will be skipped in matched filtering.")

# --- 5. Matched Filtering ---
print("\n=== 5. Matched Filtering ===")
results = []
for det in detectors:
    print(f"\nProcessing detector {det}...")
    try:
        strain_gwpy = filtered_data[det]
        delta_t = 1.0 / strain_gwpy.sample_rate.value
        strain_arr = strain_gwpy.value
        strain_ts = PyCBC_TimeSeries(strain_arr, delta_t=delta_t)
        psd = psds[det]
        if psd is None:
            print(f"  Skipping {det}: PSD unavailable.")
            continue
    except Exception as e:
        print(f"  Error preparing strain for {det}: {e}")
        continue

    for tpl_key, tpl_arr in templates.items():
        m1, m2 = tpl_key
        try:
            tpl_ts = PyCBC_TimeSeries(tpl_arr, delta_t=delta_t)
            snr = matched_filter(tpl_ts, strain_ts, psd=psd, low_frequency_cutoff=20)
            if len(snr) == 0:
                print(f"  Empty SNR for (m1={m1}, m2={m2}) in {det}, skipping.")
                continue
            crop_samples = int(0.2 / delta_t)
            if len(snr) > 2 * crop_samples:
                snr_cropped = snr.crop(crop_samples, crop_samples)
            else:
                snr_cropped = snr
            if len(snr_cropped) == 0:
                print(f"  SNR series empty after cropping for (m1={m1}, m2={m2}) in {det}, skipping.")
                continue
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            results.append({
                'detector': det,
                'm1': m1,
                'm2': m2,
                'max_abs_snr': float(max_snr),
                'max_time': float(max_time)
            })
            print(f"  {det} (m1={m1}, m2={m2}): max|SNR|={max_snr:.2f} at t={max_time:.3f}")
        except Exception as e:
            print(f"  Error for (m1={m1}, m2={m2}) in {det}: {e}")

if len(results) == 0:
    print("Warning: No matched filter results found. Check earlier steps.")

# --- 6. Output Saving ---
print("\n=== 6. Saving Results to HDF5 ===")
output_file = "gw_analysis_results.h5"
skipped_templates = []
skipped_psds = []
skipped_snrs = []

try:
    with h5py.File(output_file, "w") as f:
        # Save templates
        grp_templates = f.create_group("templates")
        print("Saving templates...")
        for (m1, m2), tpl in templates.items():
            try:
                dset_name = f"m1_{m1}_m2_{m2}"
                grp_templates.create_dataset(dset_name, data=tpl)
            except Exception as e:
                print(f"  Failed to save template (m1={m1}, m2={m2}): {e}")
                skipped_templates.append((m1, m2))
        print(f"  Templates saved: {len(templates) - len(skipped_templates)}")
        if skipped_templates:
            print(f"  Skipped templates: {skipped_templates}")

        # Save PSDs
        grp_psds = f.create_group("psds")
        print("Saving PSDs...")
        for det, psd in psds.items():
            try:
                if psd is None:
                    raise ValueError("PSD is None")
                grp_psds.create_dataset(det, data=psd.numpy())
            except Exception as e:
                print(f"  Failed to save PSD for {det}: {e}")
                skipped_psds.append(det)
        print(f"  PSDs saved: {len(psds) - len(skipped_psds)}")
        if skipped_psds:
            print(f"  Skipped PSDs: {skipped_psds}")

        # Build summary table: m1, m2, max_snr_H1, t_H1, max_snr_L1, t_L1
        print("Building summary table...")
        snr_dict = {}
        for entry in results:
            key = (entry['m1'], entry['m2'])
            det = entry['detector']
            if key not in snr_dict:
                snr_dict[key] = {}
            snr_dict[key][det] = (entry['max_abs_snr'], entry['max_time'])

        all_keys = sorted(list(templates.keys()))
        summary_dtype = [
            ('m1', 'i4'), ('m2', 'i4'),
            ('max_snr_H1', 'f8'), ('t_H1', 'f8'),
            ('max_snr_L1', 'f8'), ('t_L1', 'f8')
        ]
        summary_table = np.zeros(len(all_keys), dtype=summary_dtype)
        for i, (m1, m2) in enumerate(all_keys):
            summary_table['m1'][i] = m1
            summary_table['m2'][i] = m2
            for det in ['H1', 'L1']:
                if (m1, m2) in snr_dict and det in snr_dict[(m1, m2)]:
                    snr_val, t_val = snr_dict[(m1, m2)][det]
                    summary_table[f'max_snr_{det}'][i] = snr_val
                    summary_table[f't_{det}'][i] = t_val
                else:
                    summary_table[f'max_snr_{det}'][i] = np.nan
                    summary_table[f't_{det}'][i] = np.nan
                    skipped_snrs.append((m1, m2, det))
        f.create_dataset("summary_table", data=summary_table)
        print(f"  Summary table saved: {len(summary_table)} entries")
        if skipped_snrs:
            print(f"  Skipped SNRs: {len(skipped_snrs)} entries (missing for some (m1, m2, det))")

    print("All results saved successfully.")
    print(f"Summary: {len(templates) - len(skipped_templates)} templates, "
          f"{len(psds) - len(skipped_psds)} PSDs, "
          f"{len(summary_table) - len(skipped_snrs)//2} complete SNR entries.")

except Exception as e:
    print(f"Error during saving: {e}")

print("\n=== Analysis Complete ===")
print(f"Results saved to {output_file}")