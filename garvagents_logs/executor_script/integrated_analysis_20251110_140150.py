# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries

# --- Task 1: Download and Preprocess H1 Strain Data ---
print("="*60)
print("TASK 1: Downloading and Preprocessing H1 Strain Data")
print("="*60)

# Parameters
ifo = 'H1'
event_gps = 1126259462.4  # GW150914 GPS time
window = 12               # seconds
f_low = 30                # Hz
f_high = 250              # Hz

# Calculate start and end times
start = event_gps - window / 2
end = event_gps + window / 2

try:
    print(f"Fetching {ifo} strain data from {start} to {end} (GPS)...")
    strain = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
    print(f"Data fetched: {strain}")
except Exception as e:
    print(f"Error fetching data: {e}")
    raise

# Whitening
try:
    print("Whitening the data...")
    strain_whitened = strain.whiten()
    print("Whitening complete.")
except Exception as e:
    print(f"Error during whitening: {e}")
    raise

# Bandpass filtering
try:
    print(f"Applying bandpass filter: {f_low}-{f_high} Hz...")
    strain_filtered = strain_whitened.bandpass(f_low, f_high)
    print("Bandpass filtering complete.")
except Exception as e:
    print(f"Error during bandpass filtering: {e}")
    raise

# Ensure consistent sample rate and length
try:
    print("Checking sample rate and data length...")
    sample_rate = strain_filtered.sample_rate.value
    n_samples = len(strain_filtered)
    expected_samples = int(window * sample_rate)
    print(f"Sample rate: {sample_rate} Hz, Data length: {n_samples} samples (expected: {expected_samples})")

    # If sample rate is not standard (e.g., 4096 Hz), resample to 4096 Hz
    target_sample_rate = 4096
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
        strain_filtered = strain_filtered.resample(target_sample_rate)
        sample_rate = strain_filtered.sample_rate.value
        print(f"Resampled. New sample rate: {sample_rate} Hz.")

    # Trim or pad to ensure exact window length
    n_samples = len(strain_filtered)
    expected_samples = int(window * sample_rate)
    if n_samples > expected_samples:
        print(f"Trimming data from {n_samples} to {expected_samples} samples...")
        strain_filtered = strain_filtered[:expected_samples]
    elif n_samples < expected_samples:
        print(f"Padding data from {n_samples} to {expected_samples} samples...")
        pad_width = expected_samples - n_samples
        # Pad with zeros at the end
        strain_filtered = strain_filtered.append_zeros(pad_width)
    print(f"Final data length: {len(strain_filtered)} samples.")

except Exception as e:
    print(f"Error ensuring consistent sample rate/length: {e}")
    raise

# Save result for further analysis
h1_strain_preprocessed = strain_filtered
print("H1 strain data is preprocessed and ready for further analysis.")

# --- Task 2: Generate and Align PyCBC Waveform Templates ---
print("\n" + "="*60)
print("TASK 2: Generating and Aligning PyCBC Waveform Templates")
print("="*60)

try:
    data_length = len(h1_strain_preprocessed)
    data_delta_t = h1_strain_preprocessed.dt.value  # GWpy TimeSeries dt is astropy Quantity
    print(f"Reference data: {data_length} samples, delta_t = {data_delta_t} s")
except Exception as e:
    print("Error accessing preprocessed strain data:", e)
    raise

# Mass range and waveform parameters
mass_range = range(10, 31)  # 10 to 30 inclusive
spin1z = 0
spin2z = 0
approximant = 'IMRPhenomD'  # or 'SEOBNRv4_opt'
f_lower = 30  # Hz, matches data bandpass
min_duration = 0.2  # seconds

templates = {}  # {(m1, m2): PyCBC_TimeSeries}

for m1 in mass_range:
    for m2 in mass_range:
        if m1 < m2:
            continue  # Only unique pairs with m1 >= m2
        try:
            # Generate waveform
            hp, _ = get_td_waveform(approximant=approximant,
                                    mass1=m1, mass2=m2,
                                    spin1z=spin1z, spin2z=spin2z,
                                    delta_t=data_delta_t,
                                    f_lower=f_lower)
            duration = hp.duration
            if duration < min_duration:
                print(f"Skipping (m1={m1}, m2={m2}): duration {duration:.3f}s < {min_duration}s")
                continue

            # Pad or truncate to match data length
            hp_len = len(hp)
            if hp_len < data_length:
                # Pad at the beginning (pre-merger) with zeros
                pad_width = data_length - hp_len
                hp_padded = np.pad(hp.numpy(), (pad_width, 0), 'constant')
                hp_final = PyCBC_TimeSeries(hp_padded, delta_t=data_delta_t)
                print(f"Template (m1={m1}, m2={m2}): padded from {hp_len} to {data_length} samples")
            elif hp_len > data_length:
                # Truncate from the start (keep merger at end)
                hp_final = PyCBC_TimeSeries(hp.numpy()[-data_length:], delta_t=data_delta_t)
                print(f"Template (m1={m1}, m2={m2}): truncated from {hp_len} to {data_length} samples")
            else:
                hp_final = PyCBC_TimeSeries(hp.numpy(), delta_t=data_delta_t)
                print(f"Template (m1={m1}, m2={m2}): matches data length ({data_length} samples)")

            # Store in dictionary
            templates[(m1, m2)] = hp_final

        except Exception as e:
            print(f"Error generating template for (m1={m1}, m2={m2}): {e}")

print(f"Generated {len(templates)} templates matching data length and delta_t.")

# --- Task 3: Scale Templates, Overlay, Plot, and Save ---
print("\n" + "="*60)
print("TASK 3: Scaling Templates, Overlay Plotting, and Saving Results")
print("="*60)

# Ensure output directories exist
plot_dir = "template_overlays"
array_dir = "template_arrays"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(array_dir, exist_ok=True)

# Get strain data as numpy array and time axis
strain_np = h1_strain_preprocessed.value
strain_time = h1_strain_preprocessed.times.value
strain_max = np.max(np.abs(strain_np))

print(f"Maximum absolute amplitude of H1 strain: {strain_max:.4e}")

# Save the H1 strain data and time axis for reference
np.save(os.path.join(array_dir, "h1_strain.npy"), strain_np)
np.save(os.path.join(array_dir, "h1_strain_time.npy"), strain_time)

# Plot and save the H1 strain alone
try:
    plt.figure(figsize=(10, 4))
    plt.plot(strain_time, strain_np, label='H1 Strain', color='black')
    plt.xlabel("Time (s) since GPS {:.1f}".format(h1_strain_preprocessed.t0.value))
    plt.ylabel("Strain")
    plt.title("Processed H1 Strain Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "h1_strain.png"))
    plt.close()
    print("Saved H1 strain plot.")
except Exception as e:
    print(f"Error plotting/saving H1 strain: {e}")

# For each template: scale, overlay, plot, and save
for (m1, m2), template in templates.items():
    try:
        template_np = template.numpy()
        template_max = np.max(np.abs(template_np))
        if template_max == 0:
            print(f"Template (m1={m1}, m2={m2}) has zero max amplitude, skipping.")
            continue
        scale_factor = strain_max / template_max
        template_scaled = template_np * scale_factor

        # Overlay plot
        plt.figure(figsize=(10, 4))
        plt.plot(strain_time, strain_np, label='H1 Strain', color='black', alpha=0.7)
        plt.plot(strain_time, template_scaled, label=f'Template m1={m1}, m2={m2}', color='C1', alpha=0.7)
        plt.xlabel("Time (s) since GPS {:.1f}".format(h1_strain_preprocessed.t0.value))
        plt.ylabel("Strain")
        plt.title(f"Overlay: H1 Strain & Template (m1={m1}, m2={m2})")
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(plot_dir, f"overlay_m1_{m1}_m2_{m2}.png")
        plt.savefig(plot_filename)
        plt.close()

        # Save scaled template array and time axis
        np.save(os.path.join(array_dir, f"template_m1_{m1}_m2_{m2}.npy"), template_scaled)

        print(f"Saved overlay plot and template array for (m1={m1}, m2={m2})")

    except Exception as e:
        print(f"Error processing template (m1={m1}, m2={m2}): {e}")

print("All overlays and template arrays saved.")