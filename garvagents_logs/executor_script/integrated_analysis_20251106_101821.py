#!/usr/bin/env python3
# Integrated GW170608 Analysis Script

# =========================
# IMPORTS
# =========================
import numpy as np
import traceback
import os
import pickle

from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform  # FIXED: Correct import
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.conversions import chirp_mass, mass_ratio

# =========================
# PARAMETERS & CONSTANTS
# =========================
# GW170608 event parameters
gps_center = 1180922494.5
duration = 64  # seconds (±32s)
start_time = gps_center - duration / 2
end_time = gps_center + duration / 2
detectors = ['H1', 'L1']

# Preprocessing
low_freq = 30
high_freq = 300
whiten_pad = 8  # seconds

# Template bank
m1_vals = np.arange(10, 15, 1)  # 10, 11, 12, 13, 14
m2_vals = np.arange(7, 12, 1)   # 7, 8, 9, 10, 11
spin_vals = [-0.3, 0.0, 0.3]
f_lower = 30
f_upper = 300
delta_t = 1.0 / 4096
template_duration = 16  # seconds

# Published GW170608 parameters (for comparison)
GW170608_params = {
    'chirp_mass': 7.9,  # solar masses (approximate)
    'mass_ratio': 0.66  # m2/m1 (approximate)
}

# Output directories
output_dir = "gw170608_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. DATA LOADING
# =========================
print("="*60)
print("STEP 1: Downloading LIGO H1 and L1 strain data for GW170608")
print("="*60)

strain_data = {}
for det in detectors:
    print(f"Attempting to fetch data for {det} from {start_time} to {end_time}...")
    try:
        ts = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        if ts is None or len(ts) == 0:
            print(f"Error: No data returned for {det}.")
            strain_data[det] = None
        else:
            print(f"Successfully fetched data for {det}. Data length: {len(ts)} samples.")
            strain_data[det] = ts
    except Exception as e:
        print(f"Failed to fetch data for {det}: {e}")
        strain_data[det] = None

# Save raw data
with open(os.path.join(output_dir, "strain_data.pkl"), "wb") as f:
    pickle.dump(strain_data, f)

# Check for critical data availability
if any(strain_data[det] is None for det in detectors):
    print("Critical error: Missing strain data for one or more detectors. Aborting analysis.")
    exit(1)

# =========================
# 2. PREPROCESSING
# =========================
print("\n" + "="*60)
print("STEP 2: Preprocessing (bandpass filtering and whitening)")
print("="*60)

preprocessed_data = {}
for det in detectors:
    print(f"\nProcessing {det} data...")
    ts = strain_data.get(det)
    if ts is None:
        print(f"Warning: No data available for {det}, skipping.")
        preprocessed_data[det] = None
        continue
    try:
        # Bandpass filter
        print(f"Applying {low_freq}-{high_freq} Hz bandpass filter to {det}...")
        ts_bp = ts.bandpass(low_freq, high_freq)
        # Whitening with padding
        print(f"Whitening {det} data with {whiten_pad}s padding...")
        ts_white = ts_bp.whiten(pad=whiten_pad)
        # Trim padding
        ts_white = ts_white.crop(ts.t0.value, ts.t1.value)
        print(f"{det} preprocessing complete. Data length: {len(ts_white)} samples.")
        preprocessed_data[det] = ts_white
    except Exception as e:
        print(f"Error processing {det}: {e}")
        preprocessed_data[det] = None

# Save preprocessed data
with open(os.path.join(output_dir, "preprocessed_data.pkl"), "wb") as f:
    pickle.dump(preprocessed_data, f)

if any(preprocessed_data[det] is None for det in detectors):
    print("Critical error: Preprocessing failed for one or more detectors. Aborting analysis.")
    exit(1)

# =========================
# 3. TEMPLATE BANK GENERATION
# =========================
print("\n" + "="*60)
print("STEP 3: Generating reduced template bank (IMRPhenomPv2)")
print("="*60)

template_bank = []
failed_templates = []

for m1 in m1_vals:
    for m2 in m2_vals:
        if m2 > m1:
            continue  # Ensure m1 >= m2
        for spin1z in spin_vals:
            for spin2z in spin_vals:
                params = {
                    'approximant': 'IMRPhenomPv2',
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1z,
                    'spin2z': spin2z,
                    'delta_t': delta_t,
                    'f_lower': f_lower,
                    'f_final': f_upper,
                    'duration': template_duration
                }
                try:
                    hp, hc = get_td_waveform(**params)  # FIXED: Use get_td_waveform
                    template_bank.append({
                        'mass1': m1,
                        'mass2': m2,
                        'spin1z': spin1z,
                        'spin2z': spin2z,
                        'hp': hp,
                        'hc': hc
                    })
                    print(f"Generated template: m1={m1}, m2={m2}, spin1z={spin1z}, spin2z={spin2z}")
                except Exception as e:
                    print(f"Failed: m1={m1}, m2={m2}, spin1z={spin1z}, spin2z={spin2z} | Error: {e}")
                    failed_templates.append({
                        'mass1': m1,
                        'mass2': m2,
                        'spin1z': spin1z,
                        'spin2z': spin2z,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })

print(f"\nTemplate bank generation complete. {len(template_bank)} templates generated, {len(failed_templates)} failures.")

# Save template bank and failures
with open(os.path.join(output_dir, "template_bank.pkl"), "wb") as f:
    pickle.dump(template_bank, f)
with open(os.path.join(output_dir, "failed_templates.pkl"), "wb") as f:
    pickle.dump(failed_templates, f)

if len(template_bank) == 0:
    print("Critical error: No templates generated. Aborting analysis.")
    exit(1)

# =========================
# 4. MATCHED FILTERING & ANALYSIS
# =========================
print("\n" + "="*60)
print("STEP 4: Matched filtering and analysis")
print("="*60)

results = []
failed_matches = []

# Convert GWpy TimeSeries to PyCBC TimeSeries for matched filtering
def gwpy_to_pycbc(ts):
    return PyCBC_TimeSeries(ts.value, delta_t=ts.dt.value, epoch=ts.t0.value)

pycbc_data = {}
for det in detectors:
    try:
        pycbc_data[det] = gwpy_to_pycbc(preprocessed_data[det])
    except Exception as e:
        print(f"Error converting {det} data to PyCBC TimeSeries: {e}")
        pycbc_data[det] = None

if any(pycbc_data[det] is None for det in detectors):
    print("Critical error: Could not convert preprocessed data to PyCBC TimeSeries. Aborting analysis.")
    exit(1)

print("Starting matched filtering for all templates...")

for idx, template in enumerate(template_bank):
    try:
        hp = template['hp']
        # For each detector
        for det in detectors:
            data = pycbc_data[det]
            # Resample template if needed
            if hp.delta_t != data.delta_t:
                hp = hp.resample(data.delta_t)
            # Pad or truncate template to match data length
            if len(hp) < len(data):
                hp = hp.append_zeros(len(data) - len(hp))
            elif len(hp) > len(data):
                hp = hp[:len(data)]
            # Estimate PSD from data
            psd = data.psd(4 * data.sample_rate, avg_method='median')
            psd = interpolate(psd, len(data))
            psd = inverse_spectrum_truncation(psd, int(4 * data.sample_rate))
            # Matched filter
            snr = matched_filter(hp, data, psd=psd, low_frequency_cutoff=low_freq)
            # Store SNR time series for each detector
            template[f'snr_{det}'] = snr
            # Store peak SNR and its time
            peak = abs(snr).numpy().max()
            peak_time = snr.sample_times[np.argmax(abs(snr))]
            template[f'peak_snr_{det}'] = peak
            template[f'peak_time_{det}'] = peak_time
        # Compute network SNR (quadrature sum of H1 and L1 peaks)
        net_snr = np.sqrt(template['peak_snr_H1']**2 + template['peak_snr_L1']**2)
        template['network_snr'] = net_snr
        # Compute chirp mass and mass ratio
        mc = chirp_mass(template['mass1'], template['mass2'])
        q = mass_ratio(template['mass1'], template['mass2'])
        template['chirp_mass'] = mc
        template['mass_ratio'] = q
        results.append(template)
        print(f"Template {idx+1}/{len(template_bank)}: Net SNR={net_snr:.2f} (m1={template['mass1']}, m2={template['mass2']}, spins={template['spin1z']},{template['spin2z']})")
    except Exception as e:
        print(f"Failed matched filtering for template {idx+1}: {e}")
        failed_matches.append({'template': template, 'error': str(e), 'traceback': traceback.format_exc()})

# Save matched filtering results
with open(os.path.join(output_dir, "matched_filter_results.pkl"), "wb") as f:
    pickle.dump(results, f)
with open(os.path.join(output_dir, "failed_matches.pkl"), "wb") as f:
    pickle.dump(failed_matches, f)

# Identify best-fit template (highest network SNR)
if results:
    best_template = max(results, key=lambda x: x['network_snr'])
    print("\nBest-fit template found:")
    print(f"  m1 = {best_template['mass1']} Msol")
    print(f"  m2 = {best_template['mass2']} Msol")
    print(f"  spin1z = {best_template['spin1z']}")
    print(f"  spin2z = {best_template['spin2z']}")
    print(f"  Chirp mass = {best_template['chirp_mass']:.2f} Msol")
    print(f"  Mass ratio = {best_template['mass_ratio']:.2f}")
    print(f"  Network SNR = {best_template['network_snr']:.2f}")
    print(f"  Peak SNR H1 = {best_template['peak_snr_H1']:.2f} at {best_template['peak_time_H1']}")
    print(f"  Peak SNR L1 = {best_template['peak_snr_L1']:.2f} at {best_template['peak_time_L1']}")
    # Compare to published GW170608 values
    print("\nComparison to published GW170608 values:")
    print(f"  Published chirp mass: {GW170608_params['chirp_mass']} Msol")
    print(f"  Published mass ratio: {GW170608_params['mass_ratio']}")
    print(f"  Δ Chirp mass: {best_template['chirp_mass'] - GW170608_params['chirp_mass']:.2f} Msol")
    print(f"  Δ Mass ratio: {best_template['mass_ratio'] - GW170608_params['mass_ratio']:.2f}")
    # Save best template parameters
    with open(os.path.join(output_dir, "best_template.pkl"), "wb") as f:
        pickle.dump(best_template, f)
else:
    print("No successful matched filtering results.")

# Report statistics
print("\nTemplate bank statistics:")
print(f"  Total templates attempted: {len(template_bank)}")
print(f"  Successful matched filters: {len(results)}")
print(f"  Failed matched filters: {len(failed_matches)}")

print("\nAnalysis complete. Results saved in:", output_dir)