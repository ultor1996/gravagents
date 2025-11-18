# =========================
# GW150914 H1 Data Analysis: Download, Template Generation, Overlay Visualization
# =========================

# ---- Imports ----
import os
import numpy as np
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries

# =========================
# 1. Data Loading and Filtering
# =========================

print("="*60)
print("STEP 1: Downloading and Filtering GW150914 H1 Strain Data")
print("="*60)

# Constants
GW150914_GPS = 1126259462.4
start_time = GW150914_GPS - 8
end_time = GW150914_GPS + 4
ifo = 'H1'
low_freq = 20
high_freq = 250
target_sample_rate = 4096  # PyCBC default
target_length = int((end_time - start_time) * target_sample_rate)

print(f"Downloading {ifo} strain data from {start_time} to {end_time} (GPS)...")
try:
    # Remove deprecated 'cache' argument
    strain = TimeSeries.fetch_open_data(ifo, start_time, end_time)
    print(f"Data downloaded: {len(strain)} samples at {strain.sample_rate.value} Hz")
except Exception as e:
    print(f"Error downloading data: {e}")
    raise

# Apply bandpass filter
print(f"Applying {low_freq}-{high_freq} Hz bandpass filter...")
try:
    strain_bp = strain.bandpass(low_freq, high_freq)
    print("Bandpass filter applied.")
except Exception as e:
    print(f"Error applying bandpass filter: {e}")
    raise

# Check sample rate and resample if necessary
if abs(strain_bp.sample_rate.value - target_sample_rate) > 1e-3:
    print(f"Resampling from {strain_bp.sample_rate.value} Hz to {target_sample_rate} Hz...")
    try:
        strain_bp = strain_bp.resample(target_sample_rate)
        print(f"Resampled to {strain_bp.sample_rate.value} Hz.")
    except Exception as e:
        print(f"Error during resampling: {e}")
        raise
else:
    print(f"Sample rate is already {strain_bp.sample_rate.value} Hz.")

# Ensure length consistency (trim or pad as needed)
current_length = len(strain_bp)
if current_length > target_length:
    print(f"Trimming data from {current_length} to {target_length} samples.")
    strain_bp = strain_bp[:target_length]
elif current_length < target_length:
    print(f"Padding data from {current_length} to {target_length} samples with zeros.")
    pad_width = target_length - current_length
    strain_bp = strain_bp.append_zeros(pad_width)
else:
    print(f"Data length is already {target_length} samples.")

# Save results for downstream tasks
gw150914_h1_strain = strain_bp

print("Data loading and filtering complete.")
print(f"Final data: {len(gw150914_h1_strain)} samples at {gw150914_h1_strain.sample_rate.value} Hz.")

# =========================
# 2. Template Generation and Processing
# =========================

print("\n" + "="*60)
print("STEP 2: Generating PyCBC Waveform Templates")
print("="*60)

# Extract sample rate and length from the data
target_sample_rate = gw150914_h1_strain.sample_rate.value
target_delta_t = gw150914_h1_strain.dt.value
target_length = len(gw150914_h1_strain)
target_duration = target_length * target_delta_t

print(f"Target sample rate: {target_sample_rate} Hz, length: {target_length}, duration: {target_duration:.3f} s")

# Mass grid: 10 to 30 solar masses, integer steps, m1 >= m2
masses = np.arange(10, 31, 1)
mass_pairs = [(m1, m2) for m1 in masses for m2 in masses if m1 >= m2]

templates = {}
min_template_duration = 0.2  # seconds

for idx, (m1, m2) in enumerate(mass_pairs):
    print(f"Generating template {idx+1}/{len(mass_pairs)}: m1={m1}, m2={m2}...", end=' ')
    try:
        hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                mass1=m1, mass2=m2,
                                delta_t=target_delta_t,
                                f_lower=20.0,
                                spin1z=0, spin2z=0)
    except Exception as e:
        print(f"Failed to generate waveform: {e}")
        continue

    # Only keep templates longer than 0.2 s
    if hp.duration < min_template_duration:
        print(f"Skipped (duration {hp.duration:.3f} s < {min_template_duration} s)")
        continue

    # Pad or truncate to match target length
    if len(hp) < target_length:
        pad_width = target_length - len(hp)
        hp = hp.append_zeros(pad_width)
        print(f"Padded to {target_length} samples.", end=' ')
    elif len(hp) > target_length:
        hp = hp[:target_length]
        print(f"Truncated to {target_length} samples.", end=' ')
    else:
        print("Length matches.", end=' ')

    # Convert to PyCBC TimeSeries with correct delta_t
    try:
        template_ts = PyCBC_TimeSeries(hp, delta_t=target_delta_t)
        templates[(m1, m2)] = template_ts
        print("Template stored.")
    except Exception as e:
        print(f"Failed to convert to TimeSeries: {e}")

print(f"Template generation complete. {len(templates)} templates ready for analysis.")

# =========================
# 3. Overlay Visualization and Saving
# =========================

print("\n" + "="*60)
print("STEP 3: Overlaying Templates and Saving Results")
print("="*60)

# Ensure output directories exist
plot_dir = "template_overlays"
array_dir = "template_arrays"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(array_dir, exist_ok=True)

# Use the time axis from the strain data
strain = gw150914_h1_strain
time = strain.times.value  # GWpy: use .value for numpy array
data = strain.value        # GWpy: use .value for numpy array

print(f"Overlaying {len(templates)} templates on H1 data and saving results...")

for idx, ((m1, m2), template_ts) in enumerate(templates.items()):
    try:
        template = template_ts.numpy()  # PyCBC TimeSeries: use .numpy()
        # Overlay plot
        plt.figure(figsize=(10, 4))
        plt.plot(time, data, label="H1 Strain (filtered)", color='black', linewidth=1)
        plt.plot(time, template, label=f"Template: m1={m1}, m2={m2}", alpha=0.7)
        plt.xlabel("Time (s) relative to GPS {:.1f}".format(strain.start_time.value))
        plt.ylabel("Strain")
        plt.title(f"GW150914 H1: Overlay with Template (m1={m1}, m2={m2})")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"overlay_m1_{m1}_m2_{m2}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[{idx+1}/{len(templates)}] Plot saved: {plot_path}")

        # Save template array
        array_path = os.path.join(array_dir, f"template_m1_{m1}_m2_{m2}.npy")
        np.save(array_path, template)
        print(f"[{idx+1}/{len(templates)}] Template array saved: {array_path}")

    except Exception as e:
        print(f"Error processing template (m1={m1}, m2={m2}): {e}")

print("All overlays and template arrays saved.")
print("="*60)
print("Workflow complete.")
print("="*60)