# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import os
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter

# --- Section 1: Data Loading & Preprocessing ---
print("="*60)
print("Step 1: Downloading and preprocessing GW150914 strain data (H1 and L1)...")

gps_start = 1126259454  # 8 s before merger
gps_end = 1126259466    # 4 s after merger
ifo_list = ['H1', 'L1']
lowcut = 30
highcut = 250

# Output variables
strain_H1 = None
strain_L1 = None
strain_H1_bp = None
strain_L1_bp = None
strain_H1_whiten = None
strain_L1_whiten = None
sample_rate_H1 = None
sample_rate_L1 = None
n_samples_H1 = None
n_samples_L1 = None

os.makedirs("results", exist_ok=True)

# Download data
for ifo in ifo_list:
    try:
        print(f"Fetching {ifo} data from {gps_start} to {gps_end}...")
        strain = TimeSeries.fetch_open_data(ifo, gps_start, gps_end, cache=True)
        print(f"{ifo} data fetched: {len(strain)} samples.")
        strain.write(f"results/strain_{ifo}_raw.txt")
        if ifo == 'H1':
            strain_H1 = strain
        else:
            strain_L1 = strain
    except Exception as e:
        print(f"Error fetching {ifo} data: {e}")

if strain_H1 is None or strain_L1 is None:
    raise RuntimeError("Failed to fetch strain data for both detectors. Exiting.")

# Apply bandpass filter
try:
    print("Applying bandpass filter (30-250 Hz) to H1...")
    strain_H1_bp = strain_H1.bandpass(lowcut, highcut)
    strain_H1_bp.write("results/strain_H1_bandpassed.txt")
    print("H1 bandpass filtering complete.")
except Exception as e:
    print(f"Error during H1 bandpass filtering: {e}")

try:
    print("Applying bandpass filter (30-250 Hz) to L1...")
    strain_L1_bp = strain_L1.bandpass(lowcut, highcut)
    strain_L1_bp.write("results/strain_L1_bandpassed.txt")
    print("L1 bandpass filtering complete.")
except Exception as e:
    print(f"Error during L1 bandpass filtering: {e}")

if strain_H1_bp is None or strain_L1_bp is None:
    raise RuntimeError("Bandpass filtering failed for one or both detectors. Exiting.")

# Whitening
try:
    print("Whitening H1 data...")
    strain_H1_whiten = strain_H1_bp.whiten()
    strain_H1_whiten.write("results/strain_H1_whitened.txt")
    print("H1 whitening complete.")
except Exception as e:
    print(f"Error during H1 whitening: {e}")

try:
    print("Whitening L1 data...")
    strain_L1_whiten = strain_L1_bp.whiten()
    strain_L1_whiten.write("results/strain_L1_whitened.txt")
    print("L1 whitening complete.")
except Exception as e:
    print(f"Error during L1 whitening: {e}")

if strain_H1_whiten is None or strain_L1_whiten is None:
    raise RuntimeError("Whitening failed for one or both detectors. Exiting.")

# Extract sampling rate and number of samples
try:
    sample_rate_H1 = strain_H1.sample_rate.value
    n_samples_H1 = len(strain_H1)
    print(f"H1: Sample rate = {sample_rate_H1} Hz, Number of samples = {n_samples_H1}")
except Exception as e:
    print(f"Error extracting H1 sample rate/length: {e}")

try:
    sample_rate_L1 = strain_L1.sample_rate.value
    n_samples_L1 = len(strain_L1)
    print(f"L1: Sample rate = {sample_rate_L1} Hz, Number of samples = {n_samples_L1}")
except Exception as e:
    print(f"Error extracting L1 sample rate/length: {e}")

if sample_rate_H1 != sample_rate_L1 or n_samples_H1 != n_samples_L1:
    raise RuntimeError("Sample rates or number of samples do not match between H1 and L1.")

sample_rate = int(sample_rate_H1)
n_samples = int(n_samples_H1)
duration = n_samples / sample_rate

# --- Section 2: Template Generation & Alignment ---
print("="*60)
print("Step 2: Generating and aligning waveform templates...")

mass_grid = np.arange(20, 31, 1)
template_waveforms = {}

for m1 in mass_grid:
    for m2 in mass_grid:
        if m2 > m1:
            continue  # avoid duplicates and unphysical cases
        key = (m1, m2)
        try:
            hp, _ = get_td_waveform(
                approximant="SEOBNRv4_opt",
                mass1=m1, mass2=m2,
                delta_t=1.0/sample_rate,
                f_lower=30,
                duration=duration + 0.4,  # add extra for cropping
                spin1z=0, spin2z=0
            )
            # Crop 0.2 s from each side if long enough
            if hp.duration > duration:
                try:
                    hp = hp.crop(0.2, 0.2)
                except Exception as crop_err:
                    print(f"Warning: Could not crop template ({m1}, {m2}): {crop_err}")
            # Pad or truncate to match data length
            if len(hp) > n_samples:
                hp = hp[:n_samples]
            elif len(hp) < n_samples:
                hp = hp.append_zeros(n_samples - len(hp))
            template_waveforms[key] = hp
            print(f"Template ({m1}, {m2}) generated and aligned.")
        except Exception as e:
            print(f"Error generating template ({m1}, {m2}): {e}")

if not template_waveforms:
    raise RuntimeError("No waveform templates were generated. Exiting.")

np.save("results/template_keys.npy", np.array(list(template_waveforms.keys())))

# --- Section 3: PSD Estimation ---
print("="*60)
print("Step 3: Estimating PSDs for H1 and L1...")

def estimate_psd(strain_bp, sample_rate, label):
    data = PyCBC_TimeSeries(strain_bp.value, delta_t=1.0/sample_rate)
    seglen = 2.0  # seconds
    n_samples = len(data)
    max_seglen = n_samples / sample_rate
    while seglen > max_seglen and seglen > 0.25:
        seglen /= 2.0
    seglen = max(seglen, 0.25)
    print(f"{label}: Using segment length {seglen} s for PSD estimation.")
    try:
        psd = welch(
            data,
            seg_len=int(seglen * sample_rate),
            avg_method='median'
        )
        print(f"{label}: PSD estimation complete.")
        return psd
    except Exception as e:
        print(f"Error estimating PSD for {label}: {e}")
        return None

psd_H1 = estimate_psd(strain_H1_bp, sample_rate, "H1")
psd_L1 = estimate_psd(strain_L1_bp, sample_rate, "L1")

if psd_H1 is None or psd_L1 is None:
    raise RuntimeError("PSD estimation failed for one or both detectors. Exiting.")

np.save("results/psd_H1.npy", psd_H1.numpy())
np.save("results/psd_L1.npy", psd_L1.numpy())

# --- Section 4: Matched Filtering & Visualization ---
print("="*60)
print("Step 4: Matched filtering and visualization...")

def get_pycbc_timeseries(gwpy_ts, sample_rate):
    return PyCBC_TimeSeries(gwpy_ts.value, delta_t=1.0/sample_rate)

try:
    data_H1 = get_pycbc_timeseries(strain_H1_whiten, sample_rate)
    print("Using whitened H1 data for matched filtering.")
except Exception:
    data_H1 = get_pycbc_timeseries(strain_H1_bp, sample_rate)
    print("Using bandpassed H1 data for matched filtering.")

try:
    data_L1 = get_pycbc_timeseries(strain_L1_whiten, sample_rate)
    print("Using whitened L1 data for matched filtering.")
except Exception:
    data_L1 = get_pycbc_timeseries(strain_L1_bp, sample_rate)
    print("Using bandpassed L1 data for matched filtering.")

low_frequency_cutoff = 30.0

snr_results_H1 = {}
snr_results_L1 = {}
peak_snr_H1 = -np.inf
peak_snr_L1 = -np.inf
best_mass_H1 = None
best_mass_L1 = None
best_snr_series_H1 = None
best_snr_series_L1 = None
best_template_H1 = None
best_template_L1 = None

print("Starting matched filtering for all templates...")

for (m1, m2), template in template_waveforms.items():
    try:
        # H1
        snr_H1 = matched_filter(template, data_H1, psd=psd_H1, low_frequency_cutoff=low_frequency_cutoff)
        snr_H1 = snr_H1.crop(0.2, 0.2)
        max_snr_H1 = abs(snr_H1).numpy().max()
        snr_results_H1[(m1, m2)] = max_snr_H1
        if max_snr_H1 > peak_snr_H1:
            peak_snr_H1 = max_snr_H1
            best_mass_H1 = (m1, m2)
            best_snr_series_H1 = snr_H1
            best_template_H1 = template

        # L1
        snr_L1 = matched_filter(template, data_L1, psd=psd_L1, low_frequency_cutoff=low_frequency_cutoff)
        snr_L1 = snr_L1.crop(0.2, 0.2)
        max_snr_L1 = abs(snr_L1).numpy().max()
        snr_results_L1[(m1, m2)] = max_snr_L1
        if max_snr_L1 > peak_snr_L1:
            peak_snr_L1 = max_snr_L1
            best_mass_L1 = (m1, m2)
            best_snr_series_L1 = snr_L1
            best_template_L1 = template

        print(f"Template ({m1},{m2}): H1 peak SNR={max_snr_H1:.2f}, L1 peak SNR={max_snr_L1:.2f}")
    except Exception as e:
        print(f"Error in matched filtering for template ({m1},{m2}): {e}")

if best_mass_H1 is None or best_mass_L1 is None:
    raise RuntimeError("No best-matching template found for one or both detectors.")

print("\nBest H1 mass pair:", best_mass_H1, "Peak SNR:", peak_snr_H1)
print("Best L1 mass pair:", best_mass_L1, "Peak SNR:", peak_snr_L1)

# Save results
np.save("results/snr_results_H1.npy", snr_results_H1)
np.save("results/snr_results_L1.npy", snr_results_L1)
with open("results/best_H1.txt", "w") as f:
    f.write(f"Best H1 mass pair: {best_mass_H1}, Peak SNR: {peak_snr_H1}\n")
with open("results/best_L1.txt", "w") as f:
    f.write(f"Best L1 mass pair: {best_mass_L1}, Peak SNR: {peak_snr_L1}\n")

# Plot SNR time series for both detectors
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(best_snr_series_H1.sample_times, abs(best_snr_series_H1), label=f'H1 SNR (best: {best_mass_H1})')
plt.xlabel('Time (s)')
plt.ylabel('SNR')
plt.title('Matched Filter SNR Time Series - H1')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(best_snr_series_L1.sample_times, abs(best_snr_series_L1), label=f'L1 SNR (best: {best_mass_L1})', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('SNR')
plt.title('Matched Filter SNR Time Series - L1')
plt.legend()
plt.tight_layout()
plt.savefig("results/snr_time_series.png")
plt.show()

# Overlay best template on whitened strain for H1 and L1
def overlay_template(strain_whiten, template, sample_rate, title, fname):
    template_norm = template / np.max(np.abs(template))
    strain_norm = strain_whiten.value / np.max(np.abs(strain_whiten.value))
    t = strain_whiten.times.value
    plt.figure(figsize=(10, 4))
    plt.plot(t, strain_norm, label='Whitened Strain')
    plt.plot(t, template_norm, label='Best Template', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

overlay_template(strain_H1_whiten, best_template_H1, sample_rate, f'H1: Best Template Overlay ({best_mass_H1})', "results/overlay_H1.png")
overlay_template(strain_L1_whiten, best_template_L1, sample_rate, f'L1: Best Template Overlay ({best_mass_L1})', "results/overlay_L1.png")

print("="*60)
print("Workflow complete. Results and plots saved in the 'results/' directory.")