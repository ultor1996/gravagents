# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from astropy.time import Time
import h5py
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter

# --- Constants ---
GW150914_GPS = 1126259462.4  # GW150914 merger time in GPS seconds
WINDOW_DURATION = 12         # seconds
BANDPASS_LOW = 20            # Hz
BANDPASS_HIGH = 250          # Hz
DETECTORS = ['H1', 'L1']
OUTPUT_FILENAME = "gw_analysis_results.hdf5"
MASS_RANGE = np.arange(10, 31, 1)  # 10–30 Msun
MIN_TEMPLATE_DURATION = 0.2        # seconds
CROP_EDGE_SEC = 0.2                # seconds for SNR cropping

# --- Task 1: Data Loading and Preprocessing ---
print("\n=== Task 1: Data Loading and Preprocessing ===")
strain_data = {}
try:
    print("Setting up time window around GW150914...")
    start_time = GW150914_GPS - WINDOW_DURATION / 2
    end_time = GW150914_GPS + WINDOW_DURATION / 2

    for det in DETECTORS:
        print(f"Fetching {det} strain data from {start_time} to {end_time}...")
        ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        print(f"{det}: Data fetched. Sample rate: {ts.sample_rate.value} Hz, Length: {len(ts)} samples.")

        print(f"{det}: Applying {BANDPASS_LOW}-{BANDPASS_HIGH} Hz bandpass filter...")
        ts_bp = ts.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print(f"{det}: Bandpass filter applied.")

        strain_data[det] = ts_bp

    # Check and unify sample rates
    sr_H1 = strain_data['H1'].sample_rate.value
    sr_L1 = strain_data['L1'].sample_rate.value
    print(f"H1 sample rate: {sr_H1} Hz, L1 sample rate: {sr_L1} Hz")

    if sr_H1 != sr_L1:
        target_sr = min(sr_H1, sr_L1)
        print(f"Resampling both detectors to {target_sr} Hz for consistency...")
        for det in DETECTORS:
            if strain_data[det].sample_rate.value != target_sr:
                strain_data[det] = strain_data[det].resample(target_sr)
                print(f"{det}: Resampled to {target_sr} Hz.")
    else:
        target_sr = sr_H1

    # Ensure equal lengths
    len_H1 = len(strain_data['H1'])
    len_L1 = len(strain_data['L1'])
    print(f"Post-resampling lengths: H1={len_H1}, L1={len_L1}")

    min_len = min(len_H1, len_L1)
    if len_H1 != len_L1:
        print(f"Trimming both time series to {min_len} samples for matching lengths...")
        for det in DETECTORS:
            strain_data[det] = strain_data[det][:min_len]
        print("Trimming complete.")

    print("Data loading and preprocessing complete.")

except Exception as e:
    print(f"Error during data loading/preprocessing: {e}")
    raise

# --- Task 2: Template Generation ---
print("\n=== Task 2: Template Generation ===")
templates = {}
try:
    print("Extracting reference sample rate and length from strain data...")
    ref_strain = strain_data['H1']
    sample_rate = ref_strain.sample_rate.value
    data_length = len(ref_strain)
    delta_t = 1.0 / sample_rate
    print(f"Reference sample rate: {sample_rate} Hz, data length: {data_length} samples, delta_t: {delta_t:.6f} s")

    print("Generating waveform templates for all (m1, m2) pairs...")
    for m1 in MASS_RANGE:
        for m2 in MASS_RANGE:
            if m2 > m1:
                continue  # Only consider m1 >= m2 to avoid duplicates
            try:
                hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                       mass1=m1, mass2=m2,
                                       spin1z=0, spin2z=0,
                                       delta_t=delta_t,
                                       f_lower=20.0)
                duration = hp.duration
                if duration < MIN_TEMPLATE_DURATION:
                    continue

                waveform = hp.data
                wf_len = len(waveform)
                if wf_len < data_length:
                    pad_width = data_length - wf_len
                    waveform_padded = np.pad(waveform, (pad_width, 0), 'constant')
                    templates[(m1, m2)] = waveform_padded
                elif wf_len > data_length:
                    waveform_trunc = waveform[-data_length:]
                    templates[(m1, m2)] = waveform_trunc
                else:
                    templates[(m1, m2)] = waveform

                print(f"Template (m1={m1}, m2={m2}): duration={duration:.3f}s, length={len(templates[(m1, m2)])}")

            except Exception as e:
                print(f"Error generating template for (m1={m1}, m2={m2}): {e}")

    print(f"Total templates generated: {len(templates)}")

except Exception as e:
    print(f"Error during template generation: {e}")
    raise

# --- Task 3: PSD Estimation ---
print("\n=== Task 3: PSD Estimation ===")
psds = {}

def largest_power_of_2_leq(n):
    """Return the largest power of 2 less than or equal to n."""
    return 2 ** int(np.floor(np.log2(n)))

for det in DETECTORS:
    print(f"\nEstimating PSD for {det}...")
    data = strain_data[det].value
    sample_rate = strain_data[det].sample_rate.value
    n_samples = len(data)
    min_seglen = 32

    max_seglen = n_samples // 4
    seglen = largest_power_of_2_leq(max_seglen)
    if seglen < min_seglen:
        seglen = min_seglen

    success = False
    while seglen >= min_seglen:
        print(f"  Trying segment length: {seglen} samples")
        try:
            psd = welch(strain_data[det], seg_len=seglen, avg_method='median')
            psds[det] = psd
            print(f"  PSD estimation succeeded for {det} with segment length {seglen}.")
            success = True
            break
        except Exception as e:
            print(f"  PSD estimation failed for {det} with segment length {seglen}: {e}")
            seglen = seglen // 2

    if not success:
        print(f"  PSD estimation failed for {det} after all attempts. Skipping.")

if not psds:
    print("No valid PSDs estimated. Exiting.")
    exit(1)

# --- Task 4: Matched Filtering and Output ---
print("\n=== Task 4: Matched Filtering and Output ===")
summary_rows = []

# Ensure all delta_t are identical
try:
    data_delta_t = strain_data['H1'].delta_t
    for det in DETECTORS:
        if abs(strain_data[det].delta_t - data_delta_t) > 1e-10:
            raise ValueError(f"Delta_t mismatch in {det} strain data.")
    print("All strain data and templates have consistent delta_t.")
except Exception as e:
    print(f"Delta_t consistency check failed: {e}")
    raise

try:
    with h5py.File(OUTPUT_FILENAME, "w") as f:
        # Save templates
        tpl_grp = f.create_group("templates")
        for (m1, m2), tpl in templates.items():
            tpl_grp.create_dataset(f"{m1}_{m2}", data=tpl)
        print(f"Saved {len(templates)} templates to HDF5.")

        # Save PSDs
        psd_grp = f.create_group("psds")
        for det, psd in psds.items():
            psd_grp.create_dataset(det, data=psd.data)
        print(f"Saved PSDs for detectors: {list(psds.keys())}.")

        # Matched filtering
        print("Starting matched filtering...")
        crop_samples = int(CROP_EDGE_SEC / data_delta_t)
        for det in DETECTORS:
            if det not in psds:
                print(f"Skipping {det}: no valid PSD.")
                continue
            try:
                strain_ts = PyCBC_TimeSeries(strain_data[det].value, delta_t=data_delta_t)
                psd = psds[det]
            except Exception as e:
                print(f"Error preparing data for {det}: {e}")
                continue

            for (m1, m2), tpl in templates.items():
                try:
                    tpl_ts = PyCBC_TimeSeries(tpl, delta_t=data_delta_t)
                    snr = matched_filter(tpl_ts, strain_ts, psd=psd, low_frequency_cutoff=20.0)
                    if 2 * crop_samples >= len(snr):
                        print(f"Skipping (m1={m1}, m2={m2}, {det}): template/data too short after cropping.")
                        continue
                    snr_cropped = snr.crop(CROP_EDGE_SEC, CROP_EDGE_SEC)
                    abs_snr = np.abs(snr_cropped)
                    max_idx = np.argmax(abs_snr)
                    max_snr = abs_snr[max_idx]
                    max_time = snr_cropped.sample_times[max_idx]
                    summary_rows.append({
                        "m1": m1,
                        "m2": m2,
                        "detector": det,
                        "max_snr": float(max_snr),
                        "max_time": float(max_time)
                    })
                    print(f"{det} (m1={m1}, m2={m2}): max|SNR|={max_snr:.2f} at t={max_time:.3f}")
                except Exception as e:
                    print(f"Matched filtering failed for (m1={m1}, m2={m2}, {det}): {e}")
                    continue

        # Save summary table
        if summary_rows:
            dtype = np.dtype([
                ('m1', np.float32),
                ('m2', np.float32),
                ('detector', 'S2'),
                ('max_snr', np.float32),
                ('max_time', np.float64)
            ])
            summary_arr = np.array([
                (row['m1'], row['m2'], row['detector'].encode(), row['max_snr'], row['max_time'])
                for row in summary_rows
            ], dtype=dtype)
            f.create_dataset("summary", data=summary_arr)
            print(f"Saved summary table with {len(summary_arr)} entries.")
        else:
            print("No summary entries to save.")

    print(f"All results saved to {OUTPUT_FILENAME}")

except Exception as e:
    print(f"Error during matched filtering or HDF5 output: {e}")
    raise

print("\n=== Workflow complete. ===")