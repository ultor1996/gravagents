--------------------------------------------------
# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from pycbc.catalog import Merger
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PyCBC_TimeSeries

from gwpy.timeseries import TimeSeries as GWpy_TimeSeries

# --- Section 1: Data Downloading and Filtering ---
print("="*60)
print("STEP 1: Downloading and Filtering GW150914 Strain Data")
print("="*60)
try:
    print("Fetching GW150914 GPS time from PyCBC catalog...")
    m = Merger('GW150914')
    gw150914_gps = m.time
    print(f"GW150914 GPS time: {gw150914_gps}")
except Exception as e:
    print(f"Error fetching GW150914 GPS time: {e}")
    raise

window_duration = 12  # seconds
start_time = gw150914_gps - window_duration / 2
end_time = gw150914_gps + window_duration / 2
print(f"Data window: {start_time} to {end_time} (GPS seconds)")

try:
    print("Downloading H1 strain data...")
    h1_strain = GWpy_TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    print("H1 data downloaded successfully.")
except Exception as e:
    print(f"Error downloading H1 data: {e}")
    raise

try:
    print("Downloading L1 strain data...")
    l1_strain = GWpy_TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    print("L1 data downloaded successfully.")
except Exception as e:
    print(f"Error downloading L1 data: {e}")
    raise

try:
    print("Applying 20–250 Hz bandpass filter to H1 data...")
    h1_strain_bp = h1_strain.bandpass(20, 250, filtfilt=True)
    print("Bandpass filter applied to H1 data.")
except Exception as e:
    print(f"Error filtering H1 data: {e}")
    raise

try:
    print("Applying 20–250 Hz bandpass filter to L1 data...")
    l1_strain_bp = l1_strain.bandpass(20, 250, filtfilt=True)
    print("Bandpass filter applied to L1 data.")
except Exception as e:
    print(f"Error filtering L1 data: {e}")
    raise

h1_strain_raw = h1_strain
l1_strain_raw = l1_strain
h1_strain_filtered = h1_strain_bp
l1_strain_filtered = l1_strain_bp

print("Data loading and filtering complete.\n")

# --- Section 2: Resampling and Alignment ---
print("="*60)
print("STEP 2: Resampling and Aligning H1 and L1 Data")
print("="*60)
try:
    print("Checking sample rates (delta_t) for H1 and L1...")
    h1_dt = h1_strain_filtered.dt.value
    l1_dt = l1_strain_filtered.dt.value
    print(f"H1 delta_t: {h1_dt}, L1 delta_t: {l1_dt}")

    # Step 1: Match sample rates if needed
    if not np.isclose(h1_dt, l1_dt):
        print("Sample rates differ. Resampling to the lower (slower) sample rate for both.")
        target_dt = max(h1_dt, l1_dt)
        target_rate = 1.0 / target_dt
        if not np.isclose(h1_dt, target_dt):
            print(f"Resampling H1 to {target_rate} Hz...")
            h1_strain_filtered = h1_strain_filtered.resample(target_rate)
        if not np.isclose(l1_dt, target_dt):
            print(f"Resampling L1 to {target_rate} Hz...")
            l1_strain_filtered = l1_strain_filtered.resample(target_rate)
    else:
        print("Sample rates already match.")

    # Step 2: Align start and end times
    h1_start = h1_strain_filtered.t0.value
    l1_start = l1_strain_filtered.t0.value
    h1_end = h1_strain_filtered.t1.value
    l1_end = l1_strain_filtered.t1.value

    print(f"H1 start: {h1_start}, end: {h1_end}")
    print(f"L1 start: {l1_start}, end: {l1_end}")

    aligned_start = max(h1_start, l1_start)
    aligned_end = min(h1_end, l1_end)
    print(f"Cropping both to common interval: {aligned_start} - {aligned_end}")

    h1_aligned = h1_strain_filtered.crop(aligned_start, aligned_end)
    l1_aligned = l1_strain_filtered.crop(aligned_start, aligned_end)

    # Step 3: Ensure identical lengths
    h1_len = len(h1_aligned)
    l1_len = len(l1_aligned)
    print(f"H1 length after crop: {h1_len}, L1 length after crop: {l1_len}")

    min_len = min(h1_len, l1_len)
    if h1_len != l1_len:
        print(f"Truncating both to {min_len} samples for identical length.")
        h1_aligned = h1_aligned[:min_len]
        l1_aligned = l1_aligned[:min_len]
    else:
        print("Lengths already match.")

    h1_strain_aligned = h1_aligned
    l1_strain_aligned = l1_aligned

    print("Resampling and alignment complete.\n")

except Exception as e:
    print(f"Error during resampling and alignment: {e}")
    raise

# --- Section 3: Template Generation and Preparation ---
print("="*60)
print("STEP 3: Generating and Preparing Waveform Templates")
print("="*60)
try:
    # Use H1 as reference for sample rate and length
    data_delta_t = h1_strain_aligned.dt.value
    data_length = len(h1_strain_aligned)
    data_duration = data_length * data_delta_t

    mass_range = np.arange(10, 32, 2)
    approximant = 'SEOBNRv4_opt'  # Use 'IMRPhenomD' if this fails

    template_bank = []
    template_params = []

    print("Generating waveform templates...")
    for m1 in mass_range:
        for m2 in mass_range:
            if m2 > m1:
                continue  # Only consider m1 >= m2
            try:
                hp, _ = get_td_waveform(approximant=approximant,
                                        mass1=m1, mass2=m2,
                                        spin1z=0, spin2z=0,
                                        delta_t=data_delta_t,
                                        f_lower=20.0)
            except Exception as e:
                print(f"Failed to generate waveform for m1={m1}, m2={m2} with {approximant}: {e}")
                # Try alternate approximant if available
                if approximant != 'IMRPhenomD':
                    try:
                        hp, _ = get_td_waveform(approximant='IMRPhenomD',
                                                mass1=m1, mass2=m2,
                                                spin1z=0, spin2z=0,
                                                delta_t=data_delta_t,
                                                f_lower=20.0)
                        print(f"Used IMRPhenomD for m1={m1}, m2={m2}")
                    except Exception as e2:
                        print(f"Failed with IMRPhenomD for m1={m1}, m2={m2}: {e2}")
                        continue
                else:
                    continue

            template_duration = hp.duration
            if template_duration < 0.2:
                print(f"Skipping m1={m1}, m2={m2}: duration {template_duration:.3f} s < 0.2 s")
                continue

            # Pad or truncate to match data length
            if len(hp) < data_length:
                pad_width = data_length - len(hp)
                hp = PyCBC_TimeSeries(np.pad(hp.numpy(), (0, pad_width), 'constant'),
                                      delta_t=data_delta_t, epoch=hp.start_time)
            elif len(hp) > data_length:
                hp = hp[:data_length]

            if len(hp) != data_length:
                print(f"Template for m1={m1}, m2={m2} could not be matched to data length.")
                continue

            template_bank.append(hp)
            template_params.append({'mass1': m1, 'mass2': m2, 'approximant': approximant, 'duration': template_duration})

            print(f"Template m1={m1}, m2={m2} generated, duration={template_duration:.3f}s, length={len(hp)}")

    print(f"Total templates generated: {len(template_bank)}\n")

    waveform_templates = template_bank
    waveform_template_params = template_params

except Exception as e:
    print(f"Error during template generation: {e}")
    raise

# --- Section 4: Overlay and Save Plots ---
print("="*60)
print("STEP 4: Overlaying Templates and Saving Results")
print("="*60)

output_dir = "gw_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

def overlay_and_save(detector_name, strain_data, templates, template_params):
    print(f"Plotting and saving overlays for {detector_name}...")

    # Prepare time axis (relative to start)
    time_axis = np.arange(len(strain_data)) * strain_data.dt.value

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, strain_data, label=f"{detector_name} filtered strain", color='black', linewidth=1)

    # Overlay all templates (with transparency)
    for idx, template in enumerate(templates):
        plt.plot(time_axis, template, alpha=0.5, linewidth=1,
                 label=f"Template m1={template_params[idx]['mass1']}, m2={template_params[idx]['mass2']}" if idx < 5 else None)
        # Only label the first few to avoid legend clutter

    plt.xlabel("Time (s) since segment start")
    plt.ylabel("Strain / Template amplitude")
    plt.title(f"{detector_name} Strain with Overlayed Templates")
    plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{detector_name}_strain_templates_overlay.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Save template bank as HDF5
    h5_path = os.path.join(output_dir, f"{detector_name}_template_bank.h5")
    try:
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("templates", data=np.stack([t.numpy() for t in templates]))
            param_strs = np.array([str(p) for p in template_params], dtype='S')
            f.create_dataset("params", data=param_strs)
        print(f"Saved template bank to {h5_path}")
    except Exception as e:
        print(f"Error saving template bank for {detector_name}: {e}")

try:
    # Convert GWpy TimeSeries to numpy arrays for plotting
    print("Preparing data for overlay plotting...")
    h1_data_for_plot = h1_strain_aligned.value
    l1_data_for_plot = l1_strain_aligned.value

    # Convert templates to numpy arrays for plotting
    templates_for_plot = [t for t in waveform_templates]

    # Overlay and save for H1
    overlay_and_save("H1", h1_strain_aligned, templates_for_plot, waveform_template_params)

    # Overlay and save for L1
    overlay_and_save("L1", l1_strain_aligned, templates_for_plot, waveform_template_params)

    print("Overlay, plotting, and saving complete.\n")
except Exception as e:
    print(f"Error during overlay and saving: {e}")
    raise

print("="*60)
print("All steps completed successfully.")
print("="*60)