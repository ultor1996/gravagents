# --- Imports ---
import numpy as np
import os
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.filter import matched_filter
import traceback

# --- Section 1: Load and Bandpass Filter Strain Data ---
print("="*60)
print("STEP 1: Load and Bandpass Filter Strain Data")
print("="*60)

# ---- User: Set your file paths here ----
h1_file = 'H1_strain_data.hdf5'
l1_file = 'L1_strain_data.hdf5'
h1_channel = 'H1:STRAIN'
l1_channel = 'L1:STRAIN'

gwpy_strain = {}
metadata = {}

for det, fname, chan in [('H1', h1_file, h1_channel), ('L1', l1_file, l1_channel)]:
    print(f"\nLoading {det} strain data from {fname} ...")
    try:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} not found.")
        strain = TimeSeries.read(fname, channel=chan)
        print(f"  Loaded {len(strain)} samples, duration {strain.duration.value:.2f} s")
        strain_bp = strain.bandpass(30, 250, filtfilt=True)
        print(f"  Applied Butterworth bandpass filter (30–250 Hz)")
        delta_t = strain_bp.dt.value
        n_data = len(strain_bp)
        gwpy_strain[det] = strain_bp
        metadata[det] = {'delta_t': delta_t, 'n_data': n_data}
        print(f"  {det}: delta_t = {delta_t:.6f} s, n_data = {n_data}")
    except Exception as e:
        print(f"  Error loading or processing {det}: {e}")
        traceback.print_exc()
        gwpy_strain[det] = None
        metadata[det] = None

# --- Section 2: Align and Convert to PyCBC TimeSeries ---
print("\n" + "="*60)
print("STEP 2: Align and Convert to PyCBC TimeSeries")
print("="*60)

pycbc_strain = {}

try:
    h1 = gwpy_strain.get('H1')
    l1 = gwpy_strain.get('L1')
    if h1 is None or l1 is None:
        raise ValueError("Missing GWpy TimeSeries for H1 or L1.")

    print("Checking sampling intervals (delta_t)...")
    h1_dt = h1.dt.value
    l1_dt = l1.dt.value
    if not np.isclose(h1_dt, l1_dt, rtol=1e-9, atol=1e-12):
        raise ValueError(f"Sampling intervals do not match: H1={h1_dt}, L1={l1_dt}")

    print("Aligning epochs (start times)...")
    h1_start = h1.t0.value
    l1_start = l1.t0.value
    h1_end = h1.t1.value
    l1_end = l1.t1.value

    seg_start = max(h1_start, l1_start)
    seg_end = min(h1_end, l1_end)
    if seg_end <= seg_start:
        raise ValueError(f"No overlapping time segment between H1 and L1.")

    print(f"  Overlapping segment: {seg_start} to {seg_end} (duration {seg_end-seg_start:.3f} s)")

    h1_trim = h1.crop(seg_start, seg_end)
    l1_trim = l1.crop(seg_start, seg_end)

    if len(h1_trim) != len(l1_trim):
        min_len = min(len(h1_trim), len(l1_trim))
        print(f"  Warning: Cropped lengths differ (H1={len(h1_trim)}, L1={len(l1_trim)}). Truncating to {min_len}.")
        h1_trim = h1_trim[:min_len]
        l1_trim = l1_trim[:min_len]

    print("Converting to PyCBC TimeSeries...")
    pycbc_strain['H1'] = PyCBC_TimeSeries(h1_trim.value, delta_t=h1_trim.dt.value, epoch=h1_trim.t0.value)
    pycbc_strain['L1'] = PyCBC_TimeSeries(l1_trim.value, delta_t=l1_trim.dt.value, epoch=l1_trim.t0.value)
    print(f"  H1: {len(pycbc_strain['H1'])} samples, delta_t={pycbc_strain['H1'].delta_t}, epoch={pycbc_strain['H1'].start_time}")
    print(f"  L1: {len(pycbc_strain['L1'])} samples, delta_t={pycbc_strain['L1'].delta_t}, epoch={pycbc_strain['L1'].start_time}")

except Exception as e:
    print(f"Error during GWpy to PyCBC conversion: {e}")
    traceback.print_exc()
    pycbc_strain['H1'] = None
    pycbc_strain['L1'] = None

# --- Section 3: Generate and Align Templates ---
print("\n" + "="*60)
print("STEP 3: Generate and Align SEOBNRv4 Templates")
print("="*60)

template_bank = {}
try:
    ref_strain = pycbc_strain.get('H1')
    if ref_strain is None:
        raise ValueError("Reference strain data (H1) not found.")

    delta_t = ref_strain.delta_t
    n_data = len(ref_strain)
    mass_grid = np.arange(20, 32, 2)  # 20, 22, 24, 26, 28, 30

    for m in mass_grid:
        print(f"\nGenerating SEOBNRv4 template for m1 = m2 = {m} M_sun, zero spin...")
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4",
                                    mass1=m, mass2=m,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=30)
            max_samples_1s = int(1.0 / delta_t)
            if len(hp) > max_samples_1s:
                print(f"  Cropping template to 1 s ({max_samples_1s} samples)...")
                hp = hp[-max_samples_1s:]
            if len(hp) < n_data:
                print(f"  Padding template from {len(hp)} to {n_data} samples...")
                pad = np.zeros(n_data - len(hp))
                hp_data = np.concatenate([pad, hp.numpy()])
            elif len(hp) > n_data:
                print(f"  Truncating template from {len(hp)} to {n_data} samples...")
                hp_data = hp.numpy()[-n_data:]
            else:
                hp_data = hp.numpy()
            template = PyCBC_TimeSeries(hp_data, delta_t=delta_t, epoch=ref_strain.start_time)
            template_bank[m] = template
            print(f"  Template ready: {len(template)} samples, delta_t={template.delta_t}, epoch={template.start_time}")
        except Exception as e:
            print(f"  Error generating template for mass {m}: {e}")
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

psd_dict = {}

for det, strain in pycbc_strain.items():
    print(f"\nEstimating PSD for {det} ...")
    try:
        if strain is None:
            raise ValueError(f"No strain data for {det}.")
        n_data = len(strain)
        delta_t = strain.delta_t
        duration = n_data * delta_t
        delta_f = 1.0 / duration

        seg_len_sec = 2.0
        seg_len = int(seg_len_sec / delta_t)
        while seg_len > n_data:
            seg_len_sec /= 2
            seg_len = int(seg_len_sec / delta_t)
            print(f"  Segment length too long, reducing to {seg_len_sec:.2f} s ({seg_len} samples)")
        if seg_len < 32:
            seg_len = n_data // 4
            print(f"  Segment length too short, using fallback: {seg_len} samples")

        print(f"  Using segment length: {seg_len} samples ({seg_len * delta_t:.2f} s)")

        psd = welch(strain, seg_len=seg_len, avg_method='median')
        print(f"  Raw PSD length: {len(psd)}, df={psd.delta_f}")

        n_psd = n_data // 2 + 1
        print(f"  Interpolating and trimming PSD to {n_psd} frequency bins (delta_f={delta_f:.6f})")
        psd = psd.interpolate(delta_f)
        psd = psd.trim(0, (n_psd - 1) * delta_f)
        if len(psd) != n_psd:
            print(f"  Warning: PSD length after trim is {len(psd)}, expected {n_psd}")

        psd_dict[det] = psd
        print(f"  PSD ready: {len(psd)} bins, df={psd.delta_f}")

    except Exception as e:
        print(f"  Error estimating PSD for {det}: {e}")
        traceback.print_exc()
        psd_dict[det] = None

# --- Section 5: Matched Filtering and SNR Extraction ---
print("\n" + "="*60)
print("STEP 5: Matched Filtering and SNR Extraction")
print("="*60)

snr_results = {}

for det in pycbc_strain:
    strain = pycbc_strain[det]
    psd = psd_dict.get(det)
    if strain is None or psd is None:
        print(f"\nSkipping {det}: missing strain or PSD.")
        continue

    snr_results[det] = {}
    print(f"\nMatched filtering for detector {det} ...")

    for mass, template in template_bank.items():
        if template is None:
            print(f"  Skipping mass {mass}: template missing.")
            snr_results[det][mass] = None
            continue

        try:
            print(f"  Filtering with template mass {mass} ...")
            snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=30)
            crop_samples = int(0.2 / strain.delta_t)
            if 2 * crop_samples >= len(snr):
                print(f"    Warning: Not enough samples to crop 0.2 s from each edge. Skipping cropping.")
                snr_cropped = snr
            else:
                snr_cropped = snr[crop_samples:-crop_samples]
            abs_snr = np.abs(snr_cropped)
            max_idx = np.argmax(abs_snr)
            max_snr = abs_snr[max_idx]
            max_time = snr_cropped.sample_times[max_idx]
            snr_results[det][mass] = {'max_snr': float(max_snr), 'time': float(max_time)}
            print(f"    Max |SNR|: {max_snr:.3f} at t = {max_time:.6f}")
        except Exception as e:
            print(f"    Error filtering mass {mass}: {e}")
            traceback.print_exc()
            snr_results[det][mass] = None

# --- Section 6: Save Results ---
print("\n" + "="*60)
print("Saving Results")
print("="*60)

try:
    np.savez("snr_results.npz", snr_results=snr_results)
    print("SNR results saved to 'snr_results.npz'.")
except Exception as e:
    print(f"Error saving SNR results: {e}")

print("\nWorkflow complete.")
    import numpy as np
    snr_results = np.load("snr_results.npz", allow_pickle=True)['snr_results'].item()
    print(snr_results)