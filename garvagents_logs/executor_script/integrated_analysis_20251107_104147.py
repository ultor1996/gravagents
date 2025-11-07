# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter
import os

# --- Parameters ---
gps_start = 1126259462  # Example: GW150914 event start
gps_end = gps_start + 4  # 4 seconds of data

channels = {
    'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
    'L1': 'L1:GWOSC-4KHZ_R1_STRAIN'
}

output_dir = "gw_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Data Loading and Filtering ---
print("\n=== Task 1: Loading and Filtering Strain Data ===")
strain_data = {}
delta_t = {}
n_data = {}

for det in ['H1', 'L1']:
    print(f"Loading raw strain data for {det}...")
    try:
        ts = TimeSeries.get(channels[det], gps_start, gps_end, cache=True)
        print(f"Loaded {det} data: {len(ts)} samples, duration {ts.duration.value} s")
        delta_t[det] = ts.dt.value
        n_data[det] = len(ts)
        print(f"Sample rate (delta_t) for {det}: {delta_t[det]} s")
        print(f"Number of data points (n_data) for {det}: {n_data[det]}")
        print(f"Applying 30–250 Hz Butterworth bandpass filter to {det} data...")
        ts_filtered = ts.bandpass(30, 250, filtfilt=True)
        strain_data[det] = ts_filtered
        print(f"Filtering complete for {det}.\n")
        # Save filtered data for reproducibility
        ts_filtered.write(os.path.join(output_dir, f"{det}_filtered.gwf"), format='gwf')
    except Exception as e:
        print(f"Error loading or processing {det} data: {e}")
        strain_data[det] = None
        delta_t[det] = None
        n_data[det] = None

# --- Task 2: Conversion to PyCBC TimeSeries ---
print("\n=== Task 2: Converting to PyCBC TimeSeries ===")
pycbc_strain = {}

try:
    print("Extracting data arrays and metadata from GWpy TimeSeries...")
    h1_data = strain_data['H1'].value
    l1_data = strain_data['L1'].value
    h1_dt = strain_data['H1'].dt.value
    l1_dt = strain_data['L1'].dt.value
    h1_epoch = float(strain_data['H1'].epoch.gps)
    l1_epoch = float(strain_data['L1'].epoch.gps)
    print(f"H1: dt={h1_dt}, epoch={h1_epoch}, n={len(h1_data)}")
    print(f"L1: dt={l1_dt}, epoch={l1_epoch}, n={len(l1_data)}")

    # Ensure matching delta_t
    if not np.isclose(h1_dt, l1_dt, atol=1e-10):
        raise ValueError(f"Sample rates do not match: H1={h1_dt}, L1={l1_dt}")

    # Ensure matching epoch
    if not np.isclose(h1_epoch, l1_epoch, atol=1e-6):
        print("Epochs do not match. Aligning to the later start time and trimming data...")
        new_epoch = max(h1_epoch, l1_epoch)
        h1_offset = int(round((new_epoch - h1_epoch) / h1_dt))
        l1_offset = int(round((new_epoch - l1_epoch) / l1_dt))
        h1_data = h1_data[h1_offset:]
        l1_data = l1_data[l1_offset:]
        h1_epoch = l1_epoch = new_epoch

    # Ensure matching lengths
    min_len = min(len(h1_data), len(l1_data))
    if len(h1_data) != len(l1_data):
        print(f"Data lengths differ (H1: {len(h1_data)}, L1: {len(l1_data)}). Trimming to {min_len} samples.")
        h1_data = h1_data[:min_len]
        l1_data = l1_data[:min_len]

    # Create PyCBC TimeSeries
    print("Creating PyCBC TimeSeries objects...")
    pycbc_strain['H1'] = PyCBC_TimeSeries(h1_data, delta_t=h1_dt, epoch=h1_epoch)
    pycbc_strain['L1'] = PyCBC_TimeSeries(l1_data, delta_t=l1_dt, epoch=l1_epoch)
    print("Conversion to PyCBC TimeSeries complete.")

    # Save for reproducibility
    np.savez(os.path.join(output_dir, "pycbc_strain.npz"),
             H1=pycbc_strain['H1'].numpy(), L1=pycbc_strain['L1'].numpy(),
             delta_t=h1_dt, epoch=h1_epoch)
except Exception as e:
    print(f"Error during conversion: {e}")
    pycbc_strain['H1'] = None
    pycbc_strain['L1'] = None

# --- Task 3: Template Generation ---
print("\n=== Task 3: Generating Waveform Templates ===")
templates = {}
try:
    data_delta_t = delta_t['H1']
    data_n_data = n_data['H1']
    mass_range = np.arange(20, 32, 2)
    print("Generating zero-spin waveform templates for component masses 20–30 M☉...")
    for m1 in mass_range:
        for m2 in mass_range:
            if m2 > m1:
                continue
            key = (m1, m2)
            try:
                print(f"Generating template for m1={m1} M☉, m2={m2} M☉...")
                hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                        mass1=m1, mass2=m2,
                                        delta_t=data_delta_t,
                                        f_lower=30,
                                        spin1z=0, spin2z=0)
                max_samples = int(1.0 / data_delta_t)
                if len(hp) > max_samples:
                    print(f"Template duration > 1 s ({len(hp)*data_delta_t:.2f} s). Cropping to 1 s.")
                    hp = hp[:max_samples]
                if len(hp) < data_n_data:
                    print(f"Padding template from {len(hp)} to {data_n_data} samples.")
                    pad_width = data_n_data - len(hp)
                    hp = np.pad(hp, (0, pad_width), 'constant')
                elif len(hp) > data_n_data:
                    print(f"Truncating template from {len(hp)} to {data_n_data} samples.")
                    hp = hp[:data_n_data]
                templates[key] = np.array(hp)
            except Exception as e:
                print(f"Error generating template for m1={m1}, m2={m2}: {e}")
                templates[key] = None
    print("Template generation complete.")
    # Save templates
    np.savez(os.path.join(output_dir, "templates.npz"), **{str(k): v for k, v in templates.items() if v is not None})
except Exception as e:
    print(f"Error in template generation: {e}")

# --- Task 4: PSD Estimation ---
print("\n=== Task 4: Estimating PSDs ===")
psds = {}
for det in ['H1', 'L1']:
    print(f"\nEstimating PSD for {det}...")
    try:
        ts = pycbc_strain[det]
        if ts is None:
            raise ValueError(f"No PyCBC TimeSeries for {det}")
        sample_rate = 1.0 / ts.delta_t
        npts = len(ts)
        seg_len_sec = 2.0
        min_psd_samples = 32
        while True:
            seg_len = int(seg_len_sec * sample_rate)
            if seg_len > npts:
                seg_len_sec /= 2
                print(f"Segment length {seg_len} > data length {npts}, halving to {seg_len_sec} s")
                if seg_len_sec < ts.delta_t * 32:
                    print("Segment length too short, using fallback n_data // 4")
                    seg_len = npts // 4
                    break
            else:
                n_psd_samples = seg_len // 2 + 1
                if n_psd_samples < min_psd_samples:
                    seg_len_sec /= 2
                    print(f"PSD would have only {n_psd_samples} samples, halving segment length to {seg_len_sec} s")
                    if seg_len_sec < ts.delta_t * 32:
                        print("Segment length too short, using fallback n_data // 4")
                        seg_len = npts // 4
                        break
                else:
                    break
        print(f"Using segment length: {seg_len} samples ({seg_len / sample_rate:.3f} s)")
        psd = welch(ts, seg_len=seg_len, avg_method='median')
        print(f"PSD estimated: length={len(psd)}, delta_f={psd.delta_f}")
        psds[det] = psd
        # Save PSD
        np.save(os.path.join(output_dir, f"{det}_psd.npy"), psd.numpy())
    except Exception as e:
        print(f"Error estimating PSD for {det}: {e}")
        psds[det] = None

# --- Task 5: Matched Filtering ---
print("\n=== Task 5: Matched Filtering ===")
from pycbc.types import TimeSeries as PyCBC_TimeSeries  # For template conversion
snr_results = {}
crop_sec = 0.2

for det in ['H1', 'L1']:
    print(f"\nMatched filtering for detector {det}...")
    snr_results[det] = {}
    try:
        strain = pycbc_strain[det]
        psd = psds[det]
        if strain is None or psd is None:
            raise ValueError(f"Missing strain or PSD for {det}")
        # Ensure strain and PSD are compatible
        if len(strain) != len(psd) * 2 - 2:
            target_len = len(psd) * 2 - 2
            print(f"Resizing strain from {len(strain)} to {target_len} to match PSD.")
            strain = strain[:target_len]
        for key, template_np in templates.items():
            if template_np is None:
                snr_results[det][key] = {'max_abs_snr': None, 'time': None}
                continue
            try:
                if len(template_np) != len(strain):
                    print(f"Resizing template for {key} from {len(template_np)} to {len(strain)}")
                    template_np = template_np[:len(strain)]
                template = PyCBC_TimeSeries(template_np, delta_t=strain.delta_t, epoch=strain.start_time)
                snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=30)
                crop_samples = int(crop_sec / strain.delta_t)
                if 2 * crop_samples >= len(snr):
                    print(f"Warning: SNR series too short to crop for {key} in {det}")
                    cropped_snr = snr
                else:
                    cropped_snr = snr[crop_samples:-crop_samples]
                abs_snr = np.abs(cropped_snr)
                max_idx = np.argmax(abs_snr)
                max_abs_snr = abs_snr[max_idx]
                max_time = cropped_snr.sample_times[max_idx]
                snr_results[det][key] = {'max_abs_snr': float(max_abs_snr), 'time': float(max_time)}
            except Exception as e:
                print(f"Error filtering template {key} for {det}: {e}")
                snr_results[det][key] = {'max_abs_snr': None, 'time': None}
    except Exception as e:
        print(f"Error in matched filtering for {det}: {e}")
        snr_results[det] = None

print("Matched filtering complete.")

# Save SNR results
import json
with open(os.path.join(output_dir, "snr_results.json"), "w") as f:
    json.dump(snr_results, f, indent=2)

print(f"\nAll results saved in directory: {output_dir}")
print("Workflow complete.")