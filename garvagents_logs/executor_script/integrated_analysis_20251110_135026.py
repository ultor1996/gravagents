# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries

# --- Section 1: Download H1 Strain Data ---
print("="*60)
print("Step 1: Downloading H1 strain data for GW150914...")
start_time = 1126259454.4  # 8 seconds before merger
end_time = 1126259466.4    # 4 seconds after merger
detector = 'H1'

h1_strain = None
try:
    h1_strain = TimeSeries.fetch_open_data(detector, start_time, end_time)
    print("Data download successful.")
except Exception as e:
    print(f"Error downloading data: {e}")

if h1_strain is None:
    raise RuntimeError("Failed to download H1 strain data. Exiting.")

# --- Section 2: Preprocessing (Whitening, Bandpass, Resampling) ---
print("="*60)
print("Step 2: Preprocessing H1 strain data (whitening, bandpass, resampling)...")
low_freq = 30
high_freq = 250
desired_duration = 12.0  # seconds
desired_sample_rate = 4096  # Hz

h1_whitened_bp = None
try:
    current_sample_rate = h1_strain.sample_rate.value
    print(f"Current sample rate: {current_sample_rate} Hz")

    # Resample if needed
    if current_sample_rate != desired_sample_rate:
        print(f"Resampling from {current_sample_rate} Hz to {desired_sample_rate} Hz...")
        h1_strain = h1_strain.resample(desired_sample_rate)
        print("Resampling complete.")

    # Trim or pad to ensure exactly 12 seconds
    print("Ensuring data is exactly 12 seconds long...")
    actual_duration = h1_strain.duration.value
    if actual_duration > desired_duration:
        h1_strain = h1_strain.crop(h1_strain.t0.value, h1_strain.t0.value + desired_duration)
        print("Data trimmed to 12 seconds.")
    elif actual_duration < desired_duration:
        pad_amount = int((desired_duration - actual_duration) * desired_sample_rate)
        padded_data = np.pad(h1_strain.value, (0, pad_amount), 'constant')
        h1_strain = TimeSeries(padded_data, sample_rate=desired_sample_rate, t0=h1_strain.t0.value)
        print("Data padded to 12 seconds.")
    else:
        print("Data is already 12 seconds long.")

    # Whitening
    print("Whitening the data...")
    h1_whitened = h1_strain.whiten()
    print("Whitening complete.")

    # Bandpass filter
    print(f"Applying bandpass filter: {low_freq}-{high_freq} Hz...")
    h1_whitened_bp = h1_whitened.bandpass(low_freq, high_freq)
    print("Bandpass filtering complete.")

except Exception as e:
    print(f"Error during preprocessing: {e}")

if h1_whitened_bp is None:
    raise RuntimeError("Failed to preprocess H1 strain data. Exiting.")

# --- Section 3: Generate PyCBC Waveform Templates ---
print("="*60)
print("Step 3: Generating PyCBC waveform templates...")
delta_t = h1_whitened_bp.dt.value
sample_rate = 1.0 / delta_t
n_samples = int(desired_duration * sample_rate)
mass_range = np.arange(10, 32, 2)  # 10, 12, ..., 30
approximant = 'IMRPhenomD'
f_lower = 30.0  # Hz

templates = []
template_params = []

for m1 in mass_range:
    for m2 in mass_range:
        if m2 > m1:
            continue  # Only unique (m1 >= m2) combinations
        try:
            hp, _ = get_td_waveform(approximant=approximant,
                                    mass1=m1, mass2=m2,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=f_lower)
            if hp.duration < 0.2:
                print(f"Skipping m1={m1}, m2={m2}: duration {hp.duration:.3f} s < 0.2 s")
                continue

            # Pad or truncate to desired length (12 s)
            if len(hp) < n_samples:
                pad = n_samples - len(hp)
                hp_padded = np.pad(hp.numpy(), (pad, 0), 'constant')
            else:
                hp_padded = hp.numpy()[-n_samples:]

            template_ts = PyCBC_TimeSeries(hp_padded, delta_t=delta_t)
            templates.append(template_ts)
            template_params.append({'mass1': m1, 'mass2': m2})
            print(f"Template m1={m1}, m2={m2} generated, duration: {hp.duration:.3f} s")
        except Exception as e:
            print(f"Error generating template for m1={m1}, m2={m2}: {e}")

print(f"Generated {len(templates)} templates.")

if len(templates) == 0:
    raise RuntimeError("No valid templates generated. Exiting.")

# --- Section 4: Visualization and Saving ---
print("="*60)
print("Step 4: Visualizing and saving results...")

# Output directories
plot_dir = "template_overlays"
os.makedirs(plot_dir, exist_ok=True)
data_dir = "template_data"
os.makedirs(data_dir, exist_ok=True)

# Save processed H1 strain data as plot and numpy arrays
try:
    print("Saving processed H1 strain plot and array...")
    times = h1_whitened_bp.times.value
    strain = h1_whitened_bp.value

    plt.figure(figsize=(10, 4))
    plt.plot(times, strain, label="H1 Strain", color='black')
    plt.xlabel("Time (s)")
    plt.ylabel("Strain (whitened, bandpassed)")
    plt.title("Processed H1 Strain Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "h1_strain_processed.png"))
    plt.close()

    np.save(os.path.join(data_dir, "h1_strain_processed.npy"), strain)
    np.save(os.path.join(data_dir, "h1_strain_times.npy"), times)
    print("H1 strain plot and array saved.")
except Exception as e:
    print(f"Error saving H1 strain plot/array: {e}")

# Save all templates as .npy and in a single HDF5 file
try:
    print("Saving all templates as .npy and .hdf5...")
    h5_path = os.path.join(data_dir, "templates.hdf5")
    with h5py.File(h5_path, "w") as h5f:
        for idx, (template, params) in enumerate(zip(templates, template_params)):
            m1 = params['mass1']
            m2 = params['mass2']
            template_arr = template.numpy()
            npy_name = f"template_m1_{m1}_m2_{m2}.npy"
            np.save(os.path.join(data_dir, npy_name), template_arr)
            dset_name = f"m1_{m1}_m2_{m2}"
            h5f.create_dataset(dset_name, data=template_arr)
    print("All templates saved as .npy and in templates.hdf5.")
except Exception as e:
    print(f"Error saving templates: {e}")

# Overlay each template on the H1 strain data and save the plot
try:
    print("Creating and saving overlay plots for each template...")
    for idx, (template, params) in enumerate(zip(templates, template_params)):
        m1 = params['mass1']
        m2 = params['mass2']
        plt.figure(figsize=(10, 4))
        plt.plot(times, strain, label="H1 Strain", color='black', alpha=0.7)
        plt.plot(times, template.numpy(), label=f"Template m1={m1}, m2={m2}", color='red', alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.title(f"Overlay: H1 Strain & Template (m1={m1}, m2={m2})")
        plt.legend()
        plt.tight_layout()
        plot_name = f"overlay_m1_{m1}_m2_{m2}.png"
        plt.savefig(os.path.join(plot_dir, plot_name))
        plt.close()
        if (idx + 1) % 10 == 0 or (idx + 1) == len(templates):
            print(f"Saved {idx + 1}/{len(templates)} overlay plots.")
    print("All overlay plots saved.")
except Exception as e:
    print(f"Error creating/saving overlay plots: {e}")

print("="*60)
print("Workflow complete. All results saved in 'template_overlays/' and 'template_data/'.")