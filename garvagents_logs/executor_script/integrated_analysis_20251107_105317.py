import numpy as np
snr_results = np.load("snr_results.npz", allow_pickle=True)['snr_results'].item()
print(snr_results)
# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch, interpolate
from pycbc.filter import matched_filter
import sys
import traceback

# --- Section 1: Data Loading and Filtering ---
print("="*60)
print("STEP 1: Loading and Bandpass Filtering Strain Data (GWpy)")
print("="*60)

gps_start = 1126259462  # GW150914 trigger time - 2 seconds
gps_end = 1126259466    # GW150914 trigger time + 2 seconds
detectors = ['H1', 'L1']
strain_data = {}
sample_rates = {}
n_data = {}

for det in detectors:
    print(f"\nLoading {det} strain data from {gps_start} to {gps_end}...")
    try:
        # CRITICAL FIX: Remove deprecated 'cache' argument
        ts = GWpyTimeSeries.get(f'{det}:LOSC-STRAIN', gps_start, gps_end)
        print(f"{det} data loaded. Applying Butterworth bandpass filter (30–250 Hz)...")
        ts_filtered = ts.bandpass(30, 250, filtfilt=True)
        sample_rate = ts_filtered.sample_rate.value
        n_points = len(ts_filtered)
        strain_data[det] = ts_filtered
        sample_rates[det] = sample_rate
        n_data[det] = n_points
        print(f"{det}: sample_rate = {sample_rate} Hz, n_data = {n_points}")
    except Exception as e:
        print(f"Error loading or processing {det} data: {e}")
        traceback.print_exc()
        strain_data[det] = None
        sample_rates[det] = None
        n_data[det] = None

# Save sample rates and n_data for reproducibility
np.savez("sample_rates_and_n_data.npz", sample_rates=sample_rates, n_data=n_data)

# --- Section 2: Conversion to PyCBC TimeSeries ---
print("\n" + "="*60)
print("STEP 2: Converting to PyCBC TimeSeries and Aligning")
print("="*60)

pycbc_strain = {}

try:
    h1_ts = strain_data['H1']
    l1_ts = strain_data['L1']

    if h1_ts is None or l1_ts is None:
        raise ValueError("One or both GWpy TimeSeries are missing.")

    h1_data = h1_ts.value
    l1_data = l1_ts.value
    h1_delta_t = h1_ts.dt.value
    l1_delta_t = l1_ts.dt.value
    h1_epoch = float(h1_ts.t0.value)
    l1_epoch = float(l1_ts.t0.value)

    print(f"H1: delta_t={h1_delta_t}, epoch={h1_epoch}, n_data={len(h1_data)}")
    print(f"L1: delta_t={l1_delta_t}, epoch={l1_epoch}, n_data={len(l1_data)}")

    if not np.isclose(h1_delta_t, l1_delta_t, atol=1e-10):
        raise ValueError(f"Sample spacings do not match: H1={h1_delta_t}, L1={l1_delta_t}")

    if not np.isclose(h1_epoch, l1_epoch, atol=1e-6):
        print("Aligning epochs...")
        start_epoch = max(h1_epoch, l1_epoch)
        h1_offset = int(round((start_epoch - h1_epoch) / h1_delta_t))
        l1_offset = int(round((start_epoch - l1_epoch) / l1_delta_t))
        h1_data = h1_data[h1_offset:]
        l1_data = l1_data[l1_offset:]
        h1_epoch = l1_epoch = start_epoch
        print(f"Aligned both to epoch {start_epoch} (offsets: H1={h1_offset}, L1={l1_offset})")

    min_len = min(len(h1_data), len(l1_data))
    if len(h1_data) != len(l1_data):
        print(f"Trimming data to equal length: {min_len} samples")
        h1_data = h1_data[:min_len]
        l1_data = l1_data[:min_len]

    print("Converting to PyCBC TimeSeries...")
    pycbc_strain['H1'] = PyCBC_TimeSeries(h1_data, delta_t=h1_delta_t, epoch=h1_epoch)
    pycbc_strain['L1'] = PyCBC_TimeSeries(l1_data, delta_t=h1_delta_t, epoch=h1_epoch)
    print("Conversion successful. PyCBC TimeSeries objects created for H1 and L1.")

except Exception as e:
    print(f"Error during conversion to PyCBC TimeSeries: {e}")
    traceback.print_exc()
    pycbc_strain['H1'] = None
    pycbc_strain['L1'] = None

# --- Section 3: Template Generation and Alignment ---
print("\n" + "="*60)
print("STEP 3: Generating and Aligning Waveform Templates")
print("="*60)

template_bank = {}
try:
    ref_strain = pycbc_strain['H1']
    if ref_strain is None:
        raise RuntimeError("Reference strain data is missing. Cannot proceed.")

    n_data_ref = len(ref_strain)
    delta_t_ref = ref_strain.delta_t
    epoch_ref = ref_strain.start_time

    masses = [20, 22, 24, 26, 28, 30]
    for m in masses:
        print(f"Generating SEOBNRv4 template for m1 = m2 = {m} M_sun, zero spin...")
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4",
                                    mass1=m, mass2=m,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t_ref,
                                    f_lower=30)
            max_samples_1s = int(1.0 / delta_t_ref)
            if len(hp) > max_samples_1s:
                print(f"Cropping template to 1 second ({max_samples_1s} samples)...")
                hp = hp[-max_samples_1s:]
            if len(hp) < n_data_ref:
                print(f"Padding template from {len(hp)} to {n_data_ref} samples...")
                pad_width = n_data_ref - len(hp)
                hp_data = np.pad(hp.numpy(), (pad_width, 0), 'constant')
            elif len(hp) > n_data_ref:
                print(f"Truncating template from {len(hp)} to {n_data_ref} samples...")
                hp_data = hp.numpy()[-n_data_ref:]
            else:
                hp_data = hp.numpy()
            template = PyCBC_TimeSeries(hp_data, delta_t=delta_t_ref, epoch=epoch_ref)
            template_bank[m] = template
            print(f"Template for {m} M_sun created: {len(template)} samples, delta_t={delta_t_ref}, epoch={epoch_ref}")
        except Exception as e:
            print(f"Error generating template for {m} M_sun: {e}")
            traceback.print_exc()
            template_bank[m] = None
except Exception as e:
    print(f"Error in template generation: {e}")
    traceback.print_exc()
    template_bank = {}

# --- Section 4: PSD Estimation and Interpolation ---
print("\n" + "="*60)
print("STEP 4: PSD Estimation and Interpolation")
print("="*60)

psds = {}
delta_fs = {}
duration = {}
n_data_psd = {}

for det, ts in pycbc_strain.items():
    print(f"\nEstimating PSD for {det}...")
    try:
        if ts is None:
            raise ValueError(f"No strain data for {det}")

        n_data_psd[det] = len(ts)
        delta_t = ts.delta_t
        duration[det] = n_data_psd[det] * delta_t
        delta_f = 1.0 / duration[det]
        delta_fs[det] = delta_f

        seg_len_sec = 2.0
        seg_len = int(seg_len_sec / delta_t)
        while (seg_len > n_data_psd[det] // 2 or seg_len < 32) and seg_len_sec > delta_t:
            seg_len_sec /= 2
            seg_len = int(seg_len_sec / delta_t)
        if seg_len < 32 or seg_len > n_data_psd[det]:
            seg_len = n_data_psd[det] // 4
            print(f"Falling back to seg_len = {seg_len} samples for {det}")

        print(f"Using seg_len = {seg_len} samples ({seg_len * delta_t:.3f} s) for {det}")

        psd = welch(ts, seg_len=seg_len, avg_method='median')
        print(f"Raw PSD length: {len(psd)}, df: {psd.delta_f}")

        n_psd = n_data_psd[det] // 2 + 1
        print(f"Interpolating and trimming PSD to {n_psd} frequency bins (delta_f={delta_f:.6f})")
        psd_interp = interpolate(psd, delta_f=delta_f, length=n_psd)
        psds[det] = psd_interp
        print(f"PSD for {det} ready: {len(psd_interp)} bins, delta_f={psd_interp.delta_f}")

    except Exception as e:
        print(f"Error estimating PSD for {det}: {e}")
        traceback.print_exc()
        psds[det] = None

# --- Section 5: Matched Filtering and SNR Extraction ---
print("\n" + "="*60)
print("STEP 5: Matched Filtering and SNR Extraction")
print("="*60)

snr_results = {}

for det in pycbc_strain:
    strain = pycbc_strain[det]
    psd = psds[det]
    if strain is None or psd is None:
        print(f"Skipping {det}: missing strain or PSD.")
        continue

    snr_results[det] = {}
    print(f"\nProcessing detector {det}...")

    for m, template in template_bank.items():
        if template is None:
            print(f"  Skipping mass {m}: template missing.")
            continue

        print(f"  Matched filtering for mass {m} M_sun...")
        try:
            snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=30)
            crop_samples = int(0.2 / strain.delta_t)
            if 2 * crop_samples >= len(snr):
                raise ValueError(f"  SNR series too short to crop 0.2 s from each edge (length={len(snr)})")
            snr_cropped = snr[crop_samples:-crop_samples]
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            snr_results[det][m] = {'max_abs_snr': float(max_snr), 'time': float(max_time)}
            print(f"    Max |SNR|: {max_snr:.2f} at t={max_time:.6f}")
        except Exception as e:
            print(f"    Error for mass {m}: {e}")
            traceback.print_exc()
            snr_results[det][m] = None

# --- Section 6: Save Results ---
print("\n" + "="*60)
print("Saving Results")
print("="*60)

try:
    # Save SNR results as a numpy file for easy loading
    np.savez("snr_results.npz", snr_results=snr_results)
    print("SNR results saved to 'snr_results.npz'.")
except Exception as e:
    print(f"Error saving SNR results: {e}")

print("\nWorkflow complete.")

# --- Section 7: Load and Print Results (NO INDENTATION ERROR) ---
import numpy as np
snr_results = np.load("snr_results.npz", allow_pickle=True)['snr_results'].item()
print(snr_results)