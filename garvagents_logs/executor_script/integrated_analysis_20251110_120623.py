--------------------------------------------------
# --- GW150914 Gravitational Wave Analysis Integrated Script ---

# Imports
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch, interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import matched_filter
import numpy.lib.recfunctions as rfn

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
print("\n=== 1. Data Loading & Preprocessing ===")
event_gps = 1126259462.4  # GW150914 GPS time
window_duration = 12      # seconds
start_time = event_gps - window_duration / 2
end_time = event_gps + window_duration / 2
detectors = ['H1', 'L1']
bandpass_low = 20
bandpass_high = 250
whiten_data = True  # Set to False if whitening is not desired

strain_data = {}
sample_rates = {}
num_samples = {}

for det in detectors:
    print(f"\nFetching data for {det} from {start_time} to {end_time}...")
    try:
        ts = GwpyTimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        print(f"Successfully fetched data for {det}.")
    except Exception as e:
        print(f"Error fetching data for {det}: {e}")
        continue

    print(f"Applying {bandpass_low}-{bandpass_high} Hz bandpass filter to {det} data...")
    try:
        ts_bp = ts.bandpass(bandpass_low, bandpass_high)
        print(f"Bandpass filter applied for {det}.")
    except Exception as e:
        print(f"Error applying bandpass filter for {det}: {e}")
        continue

    if whiten_data:
        print(f"Whitening {det} data...")
        try:
            ts_proc = ts_bp.whiten()
            print(f"Whitening complete for {det}.")
        except Exception as e:
            print(f"Error whitening data for {det}: {e}")
            continue
    else:
        ts_proc = ts_bp

    # Record sample rate and number of samples
    delta_t = ts_proc.dt.value  # sample interval in seconds
    n_samples = len(ts_proc)
    print(f"{det} sample rate: {1/delta_t:.2f} Hz, number of samples: {n_samples}")

    # Store results
    strain_data[det] = ts_proc
    sample_rates[det] = 1 / delta_t
    num_samples[det] = n_samples

if len(strain_data) < 2:
    raise RuntimeError("Failed to fetch and preprocess data for both detectors. Exiting.")

# ---------------------------
# 2. Template Generation
# ---------------------------
print("\n=== 2. Template Generation ===")
ref_detector = 'H1'
if ref_detector not in sample_rates or ref_detector not in num_samples:
    raise RuntimeError(f"Reference detector {ref_detector} data missing. Exiting.")

delta_t = 1 / sample_rates[ref_detector]
data_length = num_samples[ref_detector]
mass_range = range(10, 31)  # 10 to 30 inclusive

templates = {}
skipped_templates = []

print(f"Generating templates for component masses 10–30 Msun (m1 >= m2), delta_t={delta_t:.6f}, data_length={data_length}")

for m1 in mass_range:
    for m2 in range(10, m1+1):
        key = (m1, m2)
        print(f"Generating template for m1={m1}, m2={m2}...", end=' ')
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=20)
        except Exception as e:
            print(f"FAILED (generation error: {e})")
            skipped_templates.append({'m1': m1, 'm2': m2, 'reason': f'generation error: {e}'})
            continue

        template_duration = hp.duration
        if template_duration < 0.2:
            print(f"SKIPPED (duration {template_duration:.3f} s < 0.2 s)")
            skipped_templates.append({'m1': m1, 'm2': m2, 'reason': f'duration {template_duration:.3f} s < 0.2 s'})
            continue

        template_samples = len(hp)
        if template_samples < data_length:
            pad_width = data_length - template_samples
            waveform = np.pad(hp.numpy(), (pad_width, 0), 'constant')
            print(f"OK (padded {pad_width} zeros)")
        elif template_samples > data_length:
            waveform = hp.numpy()[-data_length:]
            print(f"OK (truncated {template_samples - data_length} samples)")
        else:
            waveform = hp.numpy()
            print("OK (no padding/truncation)")

        templates[key] = waveform

print(f"\nTemplate generation complete. {len(templates)} templates generated, {len(skipped_templates)} skipped.")

if skipped_templates:
    print("\nSkipped templates during generation:")
    for entry in skipped_templates:
        print(entry)

if len(templates) == 0:
    raise RuntimeError("No valid templates generated. Exiting.")

# ---------------------------
# 3. PSD Estimation
# ---------------------------
print("\n=== 3. PSD Estimation ===")
psds = {}
segment_lengths = {}

for det in strain_data:
    print(f"\nEstimating PSD for {det}...")

    data = strain_data[det]
    if hasattr(data, 'value'):
        data_np = data.value
    elif hasattr(data, 'numpy'):
        data_np = data.numpy()
    else:
        data_np = np.asarray(data)

    fs = sample_rates[det]
    n_samples_det = len(data_np)

    seglen_samples = int(2 * fs)
    if seglen_samples > n_samples_det:
        seglen_samples = n_samples_det // 2
        print(f"  2s segment too long, using half data length: {seglen_samples} samples")
    if seglen_samples < 32:
        seglen_samples = max(32, n_samples_det // 4)
        print(f"  Segment too short, using fallback: {seglen_samples} samples")

    segment_lengths[det] = seglen_samples

    try:
        ts = TimeSeries(data_np, delta_t=1/fs)
    except Exception as e:
        print(f"  ERROR: Could not create TimeSeries for {det}: {e}")
        continue

    try:
        psd = welch(ts, seg_len=seglen_samples, avg_method='median')
        psds[det] = psd
        print(f"  PSD estimated for {det} (segment length: {seglen_samples} samples, {seglen_samples/fs:.2f} s)")
    except Exception as e:
        print(f"  ERROR: PSD estimation failed for {det}: {e}")

if len(psds) < 2:
    raise RuntimeError("PSD estimation failed for one or both detectors. Exiting.")

# ---------------------------
# 4. Matched Filtering & Summary
# ---------------------------
print("\n=== 4. Matched Filtering & Summary ===")
summary_rows = []
skipped_matched_filter = []

# Save all templates and PSDs to HDF5
h5_filename = "gw_analysis_results.hdf5"
try:
    with h5py.File(h5_filename, "w") as h5f:
        # Save templates
        temp_grp = h5f.create_group("templates")
        for (m1, m2), temp in templates.items():
            temp_grp.create_dataset(f"{m1}_{m2}", data=temp)
        # Save PSDs
        psd_grp = h5f.create_group("psds")
        for det, psd in psds.items():
            psd_grp.create_dataset(det, data=psd.numpy())
            psd_grp[det].attrs['delta_f'] = psd.delta_f
            psd_grp[det].attrs['length'] = len(psd)
    print(f"Saved templates and PSDs to {h5_filename}.")
except Exception as e:
    print(f"ERROR: Failed to save templates/PSDs to HDF5: {e}")

for (m1, m2), temp_np in templates.items():
    row = {'m1': m1, 'm2': m2}
    for det in ['H1', 'L1']:
        try:
            strain = strain_data[det]
            fs = sample_rates[det]
            n = len(strain)
            if hasattr(strain, 'value'):
                strain_np = strain.value
            elif hasattr(strain, 'numpy'):
                strain_np = strain.numpy()
            else:
                strain_np = np.asarray(strain)
            strain_ts = TimeSeries(strain_np, delta_t=1/fs)

            if len(temp_np) != n:
                print(f"Template length mismatch for {(m1, m2)} and {det}, skipping.")
                skipped_matched_filter.append({'m1': m1, 'm2': m2, 'det': det, 'reason': 'length mismatch'})
                row[f'max_snr_{det}'] = np.nan
                row[f't_{det}'] = np.nan
                continue
            template_ts = TimeSeries(temp_np, delta_t=1/fs)

            psd = psds[det]
            psd = interpolate(psd, len(strain_ts), strain_ts.delta_f)
            psd = inverse_spectrum_truncation(psd, int(4 * fs), low_frequency_cutoff=20.0)

            snr = matched_filter(template_ts, strain_ts, psd=psd, low_frequency_cutoff=20.0)
            crop_samples = int(0.2 * fs)
            snr_cropped = snr.crop(crop_samples, crop_samples)
            if len(snr_cropped) == 0:
                print(f"SNR series empty after cropping for {(m1, m2)} {det}, skipping.")
                skipped_matched_filter.append({'m1': m1, 'm2': m2, 'det': det, 'reason': 'empty SNR after cropping'})
                row[f'max_snr_{det}'] = np.nan
                row[f't_{det}'] = np.nan
                continue
            abs_snr = np.abs(snr_cropped)
            idx_max = np.argmax(abs_snr)
            max_snr = abs_snr[idx_max]
            t_max = snr_cropped.sample_times[idx_max]
            row[f'max_snr_{det}'] = max_snr
            row[f't_{det}'] = t_max
            print(f"Template {(m1, m2)} {det}: max SNR={max_snr:.2f} at t={t_max:.3f}")
        except Exception as e:
            print(f"Error in matched filtering for {(m1, m2)} {det}: {e}")
            skipped_matched_filter.append({'m1': m1, 'm2': m2, 'det': det, 'reason': str(e)})
            row[f'max_snr_{det}'] = np.nan
            row[f't_{det}'] = np.nan
    summary_rows.append(row)

# Save summary table to HDF5
summary_dtype = [
    ('m1', 'i4'), ('m2', 'i4'),
    ('max_snr_H1', 'f8'), ('t_H1', 'f8'),
    ('max_snr_L1', 'f8'), ('t_L1', 'f8')
]
summary_arr = np.array([
    (row['m1'], row['m2'],
     row.get('max_snr_H1', np.nan), row.get('t_H1', np.nan),
     row.get('max_snr_L1', np.nan), row.get('t_L1', np.nan))
    for row in summary_rows
], dtype=summary_dtype)

try:
    with h5py.File(h5_filename, "a") as h5f:
        h5f.create_dataset("summary_table", data=summary_arr)
    print(f"Saved summary table to {h5_filename}.")
except Exception as e:
    print(f"ERROR: Failed to save summary table to HDF5: {e}")

# Log skipped templates for matched filtering
if skipped_matched_filter:
    print("\nSkipped templates during matched filtering:")
    for entry in skipped_matched_filter:
        print(entry)
else:
    print("\nNo templates skipped during matched filtering.")

print("\n=== GW150914 Analysis Complete ===")
print(f"Results saved in {h5_filename}")