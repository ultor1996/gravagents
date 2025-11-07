# --- Imports ---
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import matched_filter
import os

# --- Section 1: Data Loading ---
print("="*60)
print("Step 1: Downloading GW150914 strain data (H1 and L1)...")
gw150914_time = 1126259462
window = 4  # seconds before and after
start_time = gw150914_time - window
end_time = gw150914_time + window

strain_H1 = None
strain_L1 = None

try:
    print(f"Fetching H1 strain data from {start_time} to {end_time}...")
    strain_H1 = TimeSeries.fetch_open_data('H1', start_time, end_time)
    print(f"H1 data fetched: {strain_H1}")
except Exception as e:
    print(f"Error fetching H1 data: {e}")

try:
    print(f"Fetching L1 strain data from {start_time} to {end_time}...")
    strain_L1 = TimeSeries.fetch_open_data('L1', start_time, end_time)
    print(f"L1 data fetched: {strain_L1}")
except Exception as e:
    print(f"Error fetching L1 data: {e}")

if strain_H1 is None or strain_L1 is None:
    raise RuntimeError("Failed to fetch strain data for both detectors. Exiting.")

# Save raw data for reproducibility
os.makedirs("results", exist_ok=True)
strain_H1.write("results/strain_H1_raw.txt")
strain_L1.write("results/strain_L1_raw.txt")

# --- Section 2: Preprocessing ---
print("="*60)
print("Step 2: Preprocessing (bandpass filter and whitening)...")

lowcut = 30
highcut = 250

strain_H1_bp = None
strain_L1_bp = None
strain_H1_whiten = None
strain_L1_whiten = None

try:
    print("Applying bandpass filter (30-250 Hz) to H1 data...")
    strain_H1_bp = strain_H1.bandpass(lowcut, highcut)
    print("H1 bandpass filtering complete.")
except Exception as e:
    print(f"Error during H1 bandpass filtering: {e}")

try:
    print("Applying bandpass filter (30-250 Hz) to L1 data...")
    strain_L1_bp = strain_L1.bandpass(lowcut, highcut)
    print("L1 bandpass filtering complete.")
except Exception as e:
    print(f"Error during L1 bandpass filtering: {e}")

if strain_H1_bp is None or strain_L1_bp is None:
    raise RuntimeError("Bandpass filtering failed for one or both detectors. Exiting.")

try:
    print("Whitening H1 data...")
    strain_H1_whiten = strain_H1_bp.whiten()
    print("H1 whitening complete.")
except Exception as e:
    print(f"Error during H1 whitening: {e}")

try:
    print("Whitening L1 data...")
    strain_L1_whiten = strain_L1_bp.whiten()
    print("L1 whitening complete.")
except Exception as e:
    print(f"Error during L1 whitening: {e}")

if strain_H1_whiten is None or strain_L1_whiten is None:
    raise RuntimeError("Whitening failed for one or both detectors. Exiting.")

# Save preprocessed data
strain_H1_bp.write("results/strain_H1_bandpassed.txt")
strain_L1_bp.write("results/strain_L1_bandpassed.txt")
strain_H1_whiten.write("results/strain_H1_whitened.txt")
strain_L1_whiten.write("results/strain_L1_whitened.txt")

# --- Section 3: Template Generation & PSD Estimation ---
print("="*60)
print("Step 3: Generating waveform templates and estimating PSD...")

# Extract sample rate and duration
try:
    sample_rate = int(strain_H1_bp.sample_rate.value)
    duration = strain_H1_bp.duration.value
    print(f"Sample rate: {sample_rate} Hz, Duration: {duration} s")
except Exception as e:
    print(f"Error extracting sample rate/duration: {e}")
    sample_rate = 4096  # fallback
    duration = 8        # fallback

mass_grid = np.arange(20, 31, 1)
template_waveforms = {}

print("Generating waveform templates for mass grid (20–30 M☉, zero spin)...")
for m1 in mass_grid:
    for m2 in mass_grid:
        if m2 > m1:
            continue  # avoid duplicates and unphysical cases
        key = (m1, m2)
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    delta_t=1.0/sample_rate,
                                    f_lower=30,
                                    duration=duration,
                                    spin1z=0, spin2z=0)
            # Window and pad/truncate to match data length
            hp = hp.crop(0.2, 0.2)
            if len(hp) > int(duration * sample_rate):
                hp = hp[:int(duration * sample_rate)]
            elif len(hp) < int(duration * sample_rate):
                hp = hp.append_zeros(int(duration * sample_rate) - len(hp))
            template_waveforms[key] = hp
            print(f"Template ({m1}, {m2}) generated.")
        except Exception as e:
            print(f"Error generating template ({m1}, {m2}): {e}")

if not template_waveforms:
    raise RuntimeError("No waveform templates were generated. Exiting.")

# Save template keys for reference
np.save("results/template_keys.npy", np.array(list(template_waveforms.keys())))

# Estimate PSD for each detector using median-averaged Welch method
psd_H1 = None
psd_L1 = None
seg_len = 4  # seconds

try:
    print("Estimating PSD for H1...")
    h1_pycbc = PyCBC_TimeSeries(strain_H1_bp.value, delta_t=1.0/sample_rate)
    psd_H1 = welch(h1_pycbc, seg_len * sample_rate, avg_method='median')
    print("PSD for H1 estimated.")
except Exception as e:
    print(f"Error estimating PSD for H1: {e}")

try:
    print("Estimating PSD for L1...")
    l1_pycbc = PyCBC_TimeSeries(strain_L1_bp.value, delta_t=1.0/sample_rate)
    psd_L1 = welch(l1_pycbc, seg_len * sample_rate, avg_method='median')
    print("PSD for L1 estimated.")
except Exception as e:
    print(f"Error estimating PSD for L1: {e}")

if psd_H1 is None or psd_L1 is None:
    raise RuntimeError("PSD estimation failed for one or both detectors. Exiting.")

# Save PSDs
np.save("results/psd_H1.npy", psd_H1.numpy())
np.save("results/psd_L1.npy", psd_L1.numpy())

# --- Section 4: Matched Filtering ---
print("="*60)
print("Step 4: Matched filtering and SNR analysis...")

# Convert GWpy TimeSeries to PyCBC TimeSeries for filtering
h1_data = PyCBC_TimeSeries(strain_H1_bp.value, delta_t=1.0/sample_rate)
l1_data = PyCBC_TimeSeries(strain_L1_bp.value, delta_t=1.0/sample_rate)

results_H1 = []
results_L1 = []

for (m1, m2), template in template_waveforms.items():
    try:
        # Ensure template and data are same length
        if len(template) > len(h1_data):
            template = template[:len(h1_data)]
        elif len(template) < len(h1_data):
            template = template.append_zeros(len(h1_data) - len(template))
        
        # H1
        snr_H1 = matched_filter(template, h1_data, psd=psd_H1, low_frequency_cutoff=30)
        peak_snr_H1 = abs(snr_H1).numpy().max()
        peak_idx_H1 = abs(snr_H1).numpy().argmax()
        results_H1.append({'m1': m1, 'm2': m2, 'peak_snr': peak_snr_H1, 'peak_idx': peak_idx_H1})
        
        # L1
        snr_L1 = matched_filter(template, l1_data, psd=psd_L1, low_frequency_cutoff=30)
        peak_snr_L1 = abs(snr_L1).numpy().max()
        peak_idx_L1 = abs(snr_L1).numpy().argmax()
        results_L1.append({'m1': m1, 'm2': m2, 'peak_snr': peak_snr_L1, 'peak_idx': peak_idx_L1})
        
        print(f"Matched filter done for template ({m1}, {m2})")
    except Exception as e:
        print(f"Error in matched filtering for template ({m1}, {m2}): {e}")

if not results_H1 or not results_L1:
    raise RuntimeError("Matched filtering failed for all templates. Exiting.")

# Identify best-matching template for each detector
best_H1 = max(results_H1, key=lambda x: x['peak_snr']) if results_H1 else None
best_L1 = max(results_L1, key=lambda x: x['peak_snr']) if results_L1 else None

print("\nBest-matching template for H1:")
print(best_H1)
print("\nBest-matching template for L1:")
print(best_L1)

# Save results
np.save("results/results_H1.npy", results_H1)
np.save("results/results_L1.npy", results_L1)
with open("results/best_H1.txt", "w") as f:
    f.write(str(best_H1))
with open("results/best_L1.txt", "w") as f:
    f.write(str(best_L1))

print("="*60)
print("Workflow complete. Results saved in the 'results/' directory.")