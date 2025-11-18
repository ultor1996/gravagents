# --- Imports ---
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PycbcTimeSeries
from pycbc.psd import welch
from pycbc.filter import matched_filter

# --- Constants and Parameters ---
MERGER_GPS = 1126259462.4
WINDOW = 6  # seconds before and after merger
START_TIME = MERGER_GPS - WINDOW
END_TIME = MERGER_GPS + WINDOW
IFO_LIST = ['H1', 'L1']
BANDPASS_LOW = 20
BANDPASS_HIGH = 250
MASS_RANGE = np.arange(10, 31, 1)  # 10, 11, ..., 30
MIN_TEMPLATE_DURATION = 0.2  # seconds
CROP_SEC = 0.2  # seconds to crop from each edge in matched filtering
OUTPUT_H5 = "matched_filter_results.h5"

# --- 1. Data Loading and Preprocessing ---
print("="*60)
print("STEP 1: Downloading and preprocessing GW150914 strain data")
print("="*60)
strain_data = {}

for ifo in IFO_LIST:
    try:
        print(f"Fetching data for {ifo} from {START_TIME} to {END_TIME}...")
        strain = GwpyTimeSeries.fetch_open_data(ifo, START_TIME, END_TIME, cache=True)
        print(f"{ifo} data fetched successfully.")
        print(f"Applying {BANDPASS_LOW}-{BANDPASS_HIGH} Hz bandpass filter to {ifo}...")
        strain = strain.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
        print(f"Bandpass filter applied to {ifo}.")
        strain_data[ifo] = strain
    except Exception as e:
        print(f"Error fetching or filtering data for {ifo}: {e}")
        raise

# Ensure consistent sample rates and lengths
h1 = strain_data['H1']
l1 = strain_data['L1']

print(f"H1 sample rate: {h1.sample_rate.value} Hz, length: {len(h1)}")
print(f"L1 sample rate: {l1.sample_rate.value} Hz, length: {len(l1)}")

target_sample_rate = min(h1.sample_rate.value, l1.sample_rate.value)
if h1.sample_rate.value != l1.sample_rate.value:
    print(f"Resampling both to {target_sample_rate} Hz for consistency...")
    h1 = h1.resample(target_sample_rate)
    l1 = l1.resample(target_sample_rate)
    print("Resampling complete.")

min_length = min(len(h1), len(l1))
if len(h1) != len(l1):
    print(f"Trimming both time series to {min_length} samples for alignment...")
    h1 = h1[:min_length]
    l1 = l1[:min_length]
    print("Trimming complete.")

assert h1.sample_rate.value == l1.sample_rate.value, "Sample rates do not match after resampling."
assert len(h1) == len(l1), "Lengths do not match after trimming."

strain_data['H1'] = h1
strain_data['L1'] = l1

print("Data loading and preprocessing complete.\n")

# --- 2. Template Generation ---
print("="*60)
print("STEP 2: Generating waveform templates")
print("="*60)
data_length = len(strain_data['H1'])
sample_rate = strain_data['H1'].sample_rate.value
delta_t = 1.0 / sample_rate

template_bank = []
n_attempted = 0
n_skipped_short = 0
n_failed = 0

for m1 in MASS_RANGE:
    for m2 in MASS_RANGE:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates
        n_attempted += 1
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=20.0)
            duration = hp.duration
            if duration < MIN_TEMPLATE_DURATION:
                n_skipped_short += 1
                continue

            template = hp.data
            if len(template) < data_length:
                pad_width = data_length - len(template)
                template = np.pad(template, (pad_width, 0), 'constant')
            elif len(template) > data_length:
                template = template[-data_length:]

            template_ts = PycbcTimeSeries(template, delta_t=delta_t)
            template_bank.append({
                'm1': m1,
                'm2': m2,
                'template': template_ts
            })
            print(f"Template (m1={m1}, m2={m2}) generated, duration={duration:.3f}s, length={len(template)}")
        except Exception as e:
            print(f"Error generating template for (m1={m1}, m2={m2}): {e}")
            n_failed += 1

print(f"Total templates attempted: {n_attempted}")
print(f"Templates generated: {len(template_bank)}")
print(f"Templates skipped (duration < {MIN_TEMPLATE_DURATION}s): {n_skipped_short}")
print(f"Templates failed: {n_failed}\n")

# --- 3. PSD Estimation ---
print("="*60)
print("STEP 3: Estimating PSDs for each detector")
print("="*60)
psd_dict = {}
for ifo in IFO_LIST:
    try:
        print(f"Estimating PSD for {ifo}...")
        strain = strain_data[ifo]
        n_samples = len(strain)
        max_seg = n_samples // 4
        seg_len = 2 ** int(np.floor(np.log2(max_seg)))
        seg_len = max(seg_len, 32)
        print(f"{ifo}: Using segment length {seg_len} (samples) out of {n_samples} total samples.")
        # Convert GWpy TimeSeries to PyCBC TimeSeries for PSD estimation
        if not isinstance(strain, PycbcTimeSeries):
            strain_pycbc = PycbcTimeSeries(strain.value, delta_t=1.0/strain.sample_rate.value)
        else:
            strain_pycbc = strain
        psd = welch(strain_pycbc, seg_len=seg_len, avg_method='median')
        psd_dict[ifo] = psd
        print(f"PSD estimation for {ifo} complete. PSD length: {len(psd)}")
    except Exception as e:
        print(f"Error estimating PSD for {ifo}: {e}")
        # Do not raise; skip this detector in matched filtering
        continue

print("PSD estimation complete.\n")

# --- 4. Matched Filtering and Output ---
print("="*60)
print("STEP 4: Matched filtering and saving results")
print("="*60)
summary_rows = []

with h5py.File(OUTPUT_H5, "w") as h5f:
    for ifo in IFO_LIST:
        print(f"\nProcessing detector {ifo}...")
        strain = strain_data[ifo]
        # Convert to PyCBC TimeSeries if needed
        if not isinstance(strain, PycbcTimeSeries):
            strain = PycbcTimeSeries(strain.value, delta_t=1.0/strain.sample_rate.value)
        delta_t = strain.delta_t
        psd = psd_dict.get(ifo, None)
        if psd is None:
            print(f"  PSD missing for {ifo}, skipping detector.")
            continue

        # Save PSD
        try:
            h5f.create_dataset(f"{ifo}/psd", data=psd.numpy())
            h5f[f"{ifo}/psd"].attrs['delta_f'] = psd.delta_f
        except Exception as e:
            print(f"  Error saving PSD for {ifo}: {e}")

        # Prepare group for templates
        tmpl_grp = h5f.require_group(f"{ifo}/templates")

        for idx, tmpl_entry in enumerate(template_bank):
            m1, m2 = tmpl_entry['m1'], tmpl_entry['m2']
            template = tmpl_entry['template']
            # Ensure template is PyCBC TimeSeries with correct delta_t
            if not isinstance(template, PycbcTimeSeries):
                template = PycbcTimeSeries(template, delta_t=delta_t)
            elif template.delta_t != delta_t:
                template = PycbcTimeSeries(template.data, delta_t=delta_t)

            try:
                snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=20.0)
                crop_samples = int(CROP_SEC / delta_t)
                if 2 * crop_samples >= len(snr):
                    print(f"  Template (m1={m1}, m2={m2}): SNR series too short after cropping, skipping.")
                    continue
                snr_cropped = snr[crop_samples:-crop_samples]
                abs_snr = np.abs(snr_cropped)
                max_idx = np.argmax(abs_snr)
                max_snr = abs_snr[max_idx]
                max_time = snr_cropped.sample_times[max_idx]

                # Save template
                tmpl_name = f"m1_{m1}_m2_{m2}"
                try:
                    tmpl_grp.create_dataset(tmpl_name, data=template.numpy())
                    tmpl_grp[tmpl_name].attrs['m1'] = m1
                    tmpl_grp[tmpl_name].attrs['m2'] = m2
                    tmpl_grp[tmpl_name].attrs['delta_t'] = delta_t
                except Exception as e:
                    print(f"    Error saving template (m1={m1}, m2={m2}): {e}")

                # Record summary
                summary_rows.append((m1, m2, float(max_snr), float(max_time), ifo))
                print(f"  Template (m1={m1}, m2={m2}): max|SNR|={max_snr:.2f} at t={max_time:.3f} [{ifo}]")
            except Exception as e:
                print(f"  Error with template (m1={m1}, m2={m2}) on {ifo}: {e}")
                continue

    # Save summary table
    if summary_rows:
        summary_dtype = np.dtype([
            ('m1', 'f4'), ('m2', 'f4'), ('max_snr', 'f4'), ('time', 'f8'), ('detector', 'S2')
        ])
        summary_arr = np.array([
            (row[0], row[1], row[2], row[3], row[4].encode('utf-8')) for row in summary_rows
        ], dtype=summary_dtype)
        h5f.create_dataset("summary", data=summary_arr)
        print(f"\nSummary table saved with {len(summary_rows)} entries.")
    else:
        print("\nNo valid matched filter results to save in summary table.")

print(f"\nAll results saved to {OUTPUT_H5}")