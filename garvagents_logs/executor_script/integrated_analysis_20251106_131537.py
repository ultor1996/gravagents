# =========================
# GW170608 End-to-End Analysis Script
# =========================

# --- Imports ---
import sys
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.spectrum import FrequencySeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter

# --- Task 1: Download and Validate Data ---
gps_time = 1180922494.5
start = gps_time - 32
end = gps_time + 32
ifos = ['H1', 'L1']

strain_data = {}

print("="*60)
print("Starting download of H1 and L1 strain data for GW170608...")

try:
    for ifo in ifos:
        print(f"Fetching data for {ifo} from {start} to {end}...")
        ts = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
        strain_data[ifo] = ts
        print(f"Downloaded {ifo} data: {len(ts)} samples, sample rate {ts.sample_rate.value} Hz.")

    # Validation
    for ifo, ts in strain_data.items():
        print(f"Validating data for {ifo}...")
        arr = ts.value
        if np.any(np.isnan(arr)):
            print(f"Validation failed: NaN values found in {ifo} data.")
            raise ValueError(f"NaN values found in {ifo} data.")
        if np.any(np.isinf(arr)):
            print(f"Validation failed: Infinite values found in {ifo} data.")
            raise ValueError(f"Infinite values found in {ifo} data.")

        # Check for at least one continuous 60-second segment
        sample_rate = ts.sample_rate.value
        min_samples = int(60 * sample_rate)
        finite_mask = np.isfinite(arr)
        max_run = 0
        current_run = 0
        for val in finite_mask:
            if val:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        if max_run < min_samples:
            print(f"Validation failed: No continuous 60-second segment in {ifo} data.")
            raise ValueError(f"No continuous 60-second segment in {ifo} data.")
        print(f"{ifo} data passed validation: longest continuous segment = {max_run/sample_rate:.2f} seconds.")

    print("All data downloaded and validated successfully.")

except Exception as e:
    print(f"Data loading or validation failed: {e}")
    strain_data = None

if strain_data is None or any(strain_data[ifo] is None for ifo in ifos):
    print("Critical error: Could not download and validate strain data for all detectors. Exiting.")
    sys.exit(1)

# --- Task 2: Bandpass Filtering and PSD Computation ---
filtered_data = {}
psd_data = {}

def compute_psd_with_fallback(ts, seglen, fftlength, overlap):
    try:
        print(f"  Trying PSD: segment={seglen}s, fftlength={fftlength}, overlap={overlap*100:.0f}%")
        psd = ts.psd(
            seglen=seglen,
            fftlength=fftlength,
            overlap=overlap,
            window='hann'
        )
        arr = psd.value
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)) or np.any(arr <= 0):
            raise ValueError("PSD contains invalid (NaN/inf/non-positive) values.")
        print("  PSD computation successful.")
        return psd
    except Exception as e:
        print(f"  PSD computation failed: {e}")
        return None

print("="*60)
print("Starting bandpass filtering and PSD computation...")

for ifo, ts in strain_data.items():
    if ts is None:
        print(f"No data for {ifo}, skipping.")
        filtered_data[ifo] = None
        psd_data[ifo] = None
        continue

    try:
        print(f"Processing {ifo}...")
        filtered = ts.bandpass(30, 300)
        filtered_data[ifo] = filtered
        print(f"  Bandpass filtering complete for {ifo}.")

        psd = compute_psd_with_fallback(filtered, seglen=4, fftlength=4096, overlap=0.5)
        if psd is None:
            print(f"  Retrying PSD for {ifo} with 2-second segments...")
            psd = compute_psd_with_fallback(filtered, seglen=2, fftlength=2048, overlap=0.5)
        if psd is None:
            print(f"  All PSD attempts failed for {ifo}. Using analytical PSD as fallback.")
            freqs = np.linspace(0, filtered.sample_rate.value/2, int(filtered.size/2)+1)
            median_var = np.median(np.abs(filtered.value))**2
            analytic_psd = np.ones_like(freqs) * median_var
            psd = FrequencySeries(analytic_psd, frequencies=freqs, unit=filtered.unit**2/filtered.unit)
            print(f"  Analytical PSD generated for {ifo}.")
        else:
            arr = psd.value
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)) or np.any(arr <= 0):
                raise ValueError(f"Final PSD for {ifo} contains invalid values after all attempts.")
        psd_data[ifo] = psd
        print(f"PSD processing complete for {ifo}.")

    except Exception as e:
        print(f"Error processing {ifo}: {e}")
        filtered_data[ifo] = None
        psd_data[ifo] = None

print("Bandpass filtering and PSD computation finished.")

if any(filtered_data[ifo] is None or psd_data[ifo] is None for ifo in ifos):
    print("Critical error: Filtering or PSD failed for one or more detectors. Exiting.")
    sys.exit(1)

# --- Task 3: Whitening with Error Handling ---
whitened_data = {}

print("="*60)
print("Starting whitening of filtered data using computed PSDs...")

for ifo in ifos:
    filtered = filtered_data.get(ifo)
    psd = psd_data.get(ifo)
    if filtered is None or psd is None:
        print(f"Skipping {ifo}: filtered data or PSD missing.")
        whitened_data[ifo] = None
        continue

    try:
        print(f"Whitening {ifo} data...")
        whitened = filtered.whiten(asd=psd**0.5)
        whitened_data[ifo] = whitened
        print(f"  Whitening successful for {ifo}.")
    except Exception as e:
        print(f"  Whitening failed for {ifo}: {e}")
        whitened_data[ifo] = None

print("Whitening step complete.")

# --- Task 4: Template Bank Generation with PyCBC ---
primary_masses = np.arange(10, 15, 1)
secondary_masses = np.arange(7, 12, 1)
spins = [-0.3, 0.0, 0.3]
f_lower = 30
f_upper = 300
sample_rate = 4096
durations = [16, 8]

template_bank = []

print("="*60)
print("Starting template bank generation with PyCBC...")

for m1 in primary_masses:
    for m2 in secondary_masses:
        for spin1z in spins:
            for spin2z in spins:
                params = {
                    'mass1': m1,
                    'mass2': m2,
                    'spin1z': spin1z,
                    'spin2z': spin2z,
                    'approximant': 'IMRPhenomPv2',
                    'f_lower': f_lower,
                    'delta_t': 1.0 / sample_rate
                }
                success = False
                for duration in durations:
                    try:
                        print(f"Generating template: m1={m1}, m2={m2}, spin1z={spin1z}, spin2z={spin2z}, duration={duration}s")
                        hp, hc = get_td_waveform(
                            **params,
                            duration=duration
                        )
                        template_bank.append({
                            'params': dict(params, duration=duration),
                            'hp': hp,
                            'hc': hc
                        })
                        print("  Success.")
                        success = True
                        break
                    except Exception as e:
                        print(f"  Failed with duration={duration}s: {e}")
                if not success:
                    print(f"  Skipping template: m1={m1}, m2={m2}, spin1z={spin1z}, spin2z={spin2z} (all durations failed)")

print(f"Template bank generation complete. {len(template_bank)} templates generated.")

if len(template_bank) == 0:
    print("Critical error: No templates generated. Exiting.")
    sys.exit(1)

# --- Task 5: Matched Filtering and SNR Calculation ---
snr_results = []
max_network_snr = 0
peak_template = None

print("="*60)
print("Starting matched filtering for all templates...")

for idx, template in enumerate(template_bank):
    params = template['params']
    hp = template['hp']
    template_snr = {}
    skip_template = False

    for ifo in ifos:
        # Use whitened data if available, else filtered data
        data = whitened_data.get(ifo)
        if data is None:
            data = filtered_data.get(ifo)
            if data is None:
                print(f"  {ifo}: No data available, skipping template {idx}.")
                skip_template = True
                break

        try:
            # Convert GWpy TimeSeries to PyCBC TimeSeries if needed
            if not isinstance(data, PyCBC_TimeSeries):
                # GWpy dt is astropy.Quantity, get float value
                delta_t = data.dt.value if hasattr(data.dt, 'value') else float(data.dt)
                data_pcbc = PyCBC_TimeSeries(data.value, delta_t=delta_t)
            else:
                data_pcbc = data

            # Resample template if needed to match data sample rate
            if abs(hp.delta_t - data_pcbc.delta_t) > 1e-10:
                hp_resamp = hp.resample(data_pcbc.delta_t)
            else:
                hp_resamp = hp

            # Truncate or pad template to fit data length if needed
            if len(hp_resamp) > len(data_pcbc):
                hp_resamp = hp_resamp[:len(data_pcbc)]
            elif len(hp_resamp) < len(data_pcbc):
                hp_resamp = hp_resamp.copy()
                hp_resamp.resize(len(data_pcbc))

            # Matched filter
            snr = matched_filter(hp_resamp, data_pcbc, low_frequency_cutoff=30)
            template_snr[ifo] = {
                'snr_series': snr,
                'peak_snr': abs(snr).numpy().max(),
                'peak_time': snr.sample_times[np.argmax(abs(snr))]
            }
        except Exception as e:
            print(f"  Matched filtering failed for template {idx} ({ifo}): {e}")
            skip_template = True
            break

    if skip_template or ('H1' not in template_snr) or ('L1' not in template_snr):
        print(f"  Skipping template {idx} due to errors.")
        continue

    net_snr = np.sqrt(template_snr['H1']['peak_snr']**2 + template_snr['L1']['peak_snr']**2)
    snr_results.append({
        'params': params,
        'H1': template_snr['H1'],
        'L1': template_snr['L1'],
        'network_snr': net_snr
    })

    if net_snr > max_network_snr:
        max_network_snr = net_snr
        peak_template = {
            'params': params,
            'H1': template_snr['H1'],
            'L1': template_snr['L1'],
            'network_snr': net_snr,
            'template_idx': idx
        }

    print(f"  Template {idx}: network SNR = {net_snr:.2f}")

if peak_template is not None:
    print(f"\nPeak template: idx={peak_template['template_idx']}, network SNR={peak_template['network_snr']:.2f}")
    print(f"Parameters: {peak_template['params']}")
else:
    print("No valid templates found.")

print("Matched filtering complete.")
print("="*60)

# --- Save intermediate/final results if desired (example: numpy save, pickle, etc.) ---
# For brevity, not implemented here. All results are in memory in:
# - strain_data, filtered_data, psd_data, whitened_data, template_bank, snr_results, peak_template

print("Workflow complete.")