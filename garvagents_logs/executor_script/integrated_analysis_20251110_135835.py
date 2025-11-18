# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from pycbc.types import TimeSeries as PyCBC_TimeSeries
from pycbc.filter import highpass, lowpass, resample_to_delta_t, whiten
from pycbc.waveform import get_td_waveform

# --- Parameters ---
detector = 'H1'
merger_gps = 1126259462.4
window_before = 8      # seconds before merger
window_after = 4       # seconds after merger
start_time = merger_gps - window_before
end_time = merger_gps + window_after

sample_rate = 4096.0  # Hz
duration = 12         # seconds
f_low = 30.0
f_high = 250.0

mass_range = range(10, 31)  # 10 to 30 inclusive
approximant = 'IMRPhenomD'
spin1z = 0
spin2z = 0

output_dir = "gw150914_templates"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Data Loading ---
print("="*60)
print("Step 1: Downloading H1 strain data for GW150914...")
try:
    h1_strain = GWpyTimeSeries.fetch_open_data(detector, start_time, end_time)
    print("Data successfully fetched.")
    print(f"TimeSeries length: {len(h1_strain)} samples")
    print(f"Sample rate: {h1_strain.sample_rate.value} Hz")
except Exception as e:
    print(f"Error fetching data: {e}")
    raise RuntimeError("Failed to fetch H1 strain data. Exiting.")

# --- 2. Preprocessing ---
print("="*60)
print("Step 2: Preprocessing (whitening, bandpass, resampling)...")
try:
    # Convert GWpy TimeSeries to PyCBC TimeSeries
    pycbc_strain = PyCBC_TimeSeries(h1_strain.value, delta_t=h1_strain.dt.value, epoch=h1_strain.t0.value)
    print(f"Original sample rate: {1/pycbc_strain.delta_t} Hz, length: {len(pycbc_strain)} samples")

    # Resample if needed
    if not np.isclose(1/pycbc_strain.delta_t, sample_rate):
        print(f"Resampling to {sample_rate} Hz...")
        pycbc_strain = resample_to_delta_t(pycbc_strain, 1/sample_rate)
        print(f"Resampled: {1/pycbc_strain.delta_t} Hz, length: {len(pycbc_strain)} samples")

    # Whiten the data
    print("Whitening the data...")
    pycbc_strain_whitened = whiten(pycbc_strain, sample_rate, seg_len=4, seg_stride=2)
    
    # Bandpass filter (highpass then lowpass)
    print(f"Applying bandpass filter: {f_low}-{f_high} Hz...")
    pycbc_strain_bp = highpass(pycbc_strain_whitened, f_low, sample_rate)
    pycbc_strain_bp = lowpass(pycbc_strain_bp, f_high, sample_rate)

    # Trim or pad to exact duration if necessary
    expected_samples = int(sample_rate * duration)
    if len(pycbc_strain_bp) > expected_samples:
        print("Trimming data to 12 seconds...")
        pycbc_strain_bp = pycbc_strain_bp[:expected_samples]
    elif len(pycbc_strain_bp) < expected_samples:
        print("Padding data to 12 seconds...")
        pad = expected_samples - len(pycbc_strain_bp)
        pycbc_strain_bp = pycbc_strain_bp.append_zeros(pad)

    print(f"Final sample rate: {1/pycbc_strain_bp.delta_t} Hz, length: {len(pycbc_strain_bp)} samples")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise RuntimeError("Failed during preprocessing. Exiting.")

# --- 3. Template Generation ---
print("="*60)
print("Step 3: Generating waveform templates...")
delta_t = pycbc_strain_bp.delta_t
data_length = len(pycbc_strain_bp)
data_duration = data_length * delta_t

templates = []
template_params = []
num_attempted = 0
num_skipped_short = 0
num_skipped_error = 0

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates
        num_attempted += 1
        try:
            hp, _ = get_td_waveform(approximant=approximant,
                                    mass1=m1, mass2=m2,
                                    spin1z=spin1z, spin2z=spin2z,
                                    delta_t=delta_t, f_lower=f_low)
            duration = len(hp) * delta_t
            if duration < 0.2:
                num_skipped_short += 1
                continue  # Skip short templates

            # Pad or truncate to match data length
            if len(hp) < data_length:
                pad = data_length - len(hp)
                hp = hp.copy()
                hp.prepend_zeros(pad)
            elif len(hp) > data_length:
                hp = hp[-data_length:]

            hp = PyCBC_TimeSeries(hp, delta_t=delta_t)
            if len(hp) != data_length:
                print(f"Warning: Template for (m1={m1}, m2={m2}) has length {len(hp)}, expected {data_length}")
                continue

            templates.append(hp)
            template_params.append({'mass1': m1, 'mass2': m2})
            print(f"Template (m1={m1}, m2={m2}): duration={duration:.2f}s, length={len(hp)}")
        except Exception as e:
            print(f"Error generating template for (m1={m1}, m2={m2}): {e}")
            num_skipped_error += 1

print(f"Generated {len(templates)} templates matching criteria.")
print(f"Skipped {num_skipped_short} templates (too short), {num_skipped_error} due to errors.")

if len(templates) == 0:
    raise RuntimeError("No valid templates generated. Exiting.")

# --- 4. Visualization and Saving ---
print("="*60)
print("Step 4: Visualization and saving...")

# Save processed H1 strain data as .npy
h1_strain_array = pycbc_strain_bp.numpy()
h1_strain_path = os.path.join(output_dir, "processed_H1_strain.npy")
np.save(h1_strain_path, h1_strain_array)
print(f"Processed H1 strain saved to {h1_strain_path}")

# Plot and save processed H1 strain
try:
    plt.figure(figsize=(10, 4))
    plt.plot(pycbc_strain_bp.sample_times, h1_strain_array, label="Processed H1 Strain")
    plt.xlabel("Time (s, GPS)")
    plt.ylabel("Strain (whitened, bandpassed)")
    plt.title("Processed H1 Strain Data (GW150914)")
    plt.legend()
    plt.tight_layout()
    strain_plot_path = os.path.join(output_dir, "processed_H1_strain.png")
    plt.savefig(strain_plot_path)
    plt.close()
    print(f"Processed H1 strain plot saved to {strain_plot_path}")
except Exception as e:
    print(f"Error plotting processed H1 strain: {e}")

# For each template: scale, overlay, plot, and save
template_arrays = []
for idx, (template, params) in enumerate(zip(templates, template_params)):
    try:
        max_strain = np.max(np.abs(h1_strain_array))
        max_template = np.max(np.abs(template.numpy()))
        if max_template == 0:
            print(f"Template {idx} (m1={params['mass1']}, m2={params['mass2']}) has zero amplitude, skipping.")
            continue
        scale_factor = max_strain / max_template
        scaled_template = template.numpy() * scale_factor

        # Save template array as .npy
        template_fname = f"template_m1_{params['mass1']}_m2_{params['mass2']}.npy"
        template_path = os.path.join(output_dir, template_fname)
        np.save(template_path, scaled_template)
        template_arrays.append({'params': params, 'file': template_path})

        # Overlay plot
        plt.figure(figsize=(10, 4))
        plt.plot(pycbc_strain_bp.sample_times, h1_strain_array, label="Processed H1 Strain", alpha=0.7)
        plt.plot(pycbc_strain_bp.sample_times, scaled_template, label=f"Template m1={params['mass1']} m2={params['mass2']}", alpha=0.7)
        plt.xlabel("Time (s, GPS)")
        plt.ylabel("Strain (scaled)")
        plt.title(f"Overlay: H1 Strain & Template (m1={params['mass1']}, m2={params['mass2']})")
        plt.legend()
        plt.tight_layout()
        plot_fname = f"overlay_m1_{params['mass1']}_m2_{params['mass2']}.png"
        plot_path = os.path.join(output_dir, plot_fname)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved overlay plot and template for m1={params['mass1']}, m2={params['mass2']}")
    except Exception as e:
        print(f"Error processing template {idx} (m1={params['mass1']}, m2={params['mass2']}): {e}")

print(f"All templates and plots saved in '{output_dir}'.")

# Optionally, save all template arrays and parameters as a single .npz file for convenience
try:
    all_templates_npz_path = os.path.join(output_dir, "all_templates.npz")
    np.savez(
        all_templates_npz_path,
        **{f"template_m1_{t['params']['mass1']}_m2_{t['params']['mass2']}": np.load(t['file']) for t in template_arrays}
    )
    print(f"All templates saved in compressed format to {all_templates_npz_path}")
except Exception as e:
    print(f"Error saving all templates as .npz: {e}")

print("="*60)
print("Workflow complete.")