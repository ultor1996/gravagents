# =========================
# GW150914 Data & Template Overlay Workflow
# =========================

# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PycbcTimeSeries

# --- Constants ---
MERGER_GPS = 1126259462.4
WINDOW = 12  # seconds
FREQ_LOW = 20
FREQ_HIGH = 250
DETECTORS = ['H1', 'L1']
MASS_RANGE = np.arange(10, 31, 1)
MIN_TEMPLATE_DURATION = 0.2  # seconds

OUTPUT_DIR = "template_overlay_outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
NPY_DIR = os.path.join(OUTPUT_DIR, "npy")

# --- Ensure Output Directories Exist ---
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

# =========================
# Task 1: Data Loading and Preprocessing
# =========================
print("\n=== Task 1: Downloading and Preprocessing GW150914 Data ===")
strain_data = {}

try:
    print("Starting data download for GW150914...")
    for det in DETECTORS:
        print(f"Fetching data for {det}...")
        ts = GwpyTimeSeries.fetch_open_data(
            det,
            MERGER_GPS - WINDOW/2,
            MERGER_GPS + WINDOW/2,
            verbose=True
        )
        print(f"Applying {FREQ_LOW}-{FREQ_HIGH} Hz bandpass filter to {det}...")
        ts_bp = ts.bandpass(FREQ_LOW, FREQ_HIGH)
        strain_data[det] = ts_bp
    print("Data download and filtering complete.")
except Exception as e:
    print(f"Error during data download or filtering: {e}")
    raise

# Ensure matching sample rates and lengths
try:
    print("Checking sample rates and lengths...")
    h1 = strain_data['H1']
    l1 = strain_data['L1']

    # Check sample rates
    if h1.sample_rate != l1.sample_rate:
        print(f"Sample rates differ: H1={h1.sample_rate}, L1={l1.sample_rate}")
        # Resample L1 to H1's sample rate
        l1 = l1.resample(h1.sample_rate)
        print(f"L1 resampled to {h1.sample_rate} Hz.")
    else:
        print(f"Sample rates match: {h1.sample_rate} Hz.")

    # Check lengths
    min_length = min(len(h1), len(l1))
    if len(h1) != len(l1):
        print(f"Lengths differ: H1={len(h1)}, L1={len(l1)}. Trimming to {min_length} samples.")
        h1 = h1[:min_length]
        l1 = l1[:min_length]
    else:
        print(f"Lengths match: {len(h1)} samples.")

    # Save back to output variables
    strain_data['H1'] = h1
    strain_data['L1'] = l1
    print("Sample rates and lengths are now consistent.")

except Exception as e:
    print(f"Error during sample rate/length synchronization: {e}")
    raise

# =========================
# Task 2: Generate and Prepare Templates
# =========================
print("\n=== Task 2: Generating PyCBC Waveform Templates ===")
try:
    # Use H1 as reference for sample rate and length
    data_ts = strain_data['H1']
    delta_t = float(data_ts.delta_t)
    data_length = len(data_ts)
    data_duration = data_length * delta_t
    print(f"Reference data: delta_t={delta_t}, length={data_length}, duration={data_duration:.3f}s")
except Exception as e:
    print(f"Error accessing strain data from Task 1: {e}")
    raise

templates = {}
template_count = 0
skipped_short = 0

print("Generating waveform templates...")
for m1 in MASS_RANGE:
    for m2 in MASS_RANGE:
        if m2 > m1:
            continue  # Only consider m1 >= m2 to avoid duplicates
        try:
            # Generate waveform (zero spin, SEOBNRv4_opt approximant)
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=0, spin2z=0,
                                    delta_t=delta_t,
                                    f_lower=FREQ_LOW)
            template_duration = hp.duration
            if template_duration < MIN_TEMPLATE_DURATION:
                skipped_short += 1
                continue  # Skip short templates

            # Pad or truncate to match data length
            if len(hp) < data_length:
                pad_width = data_length - len(hp)
                hp_padded = np.pad(hp.numpy(), (pad_width, 0), 'constant')
                hp_ts = PycbcTimeSeries(hp_padded, delta_t=delta_t)
            elif len(hp) > data_length:
                hp_ts = PycbcTimeSeries(hp.numpy()[-data_length:], delta_t=delta_t)
            else:
                hp_ts = hp

            templates[(m1, m2)] = hp_ts
            template_count += 1
            if template_count % 20 == 0:
                print(f"{template_count} templates generated...")

        except Exception as e:
            print(f"Error generating template for (m1={m1}, m2={m2}): {e}")

print(f"Template generation complete. {template_count} templates kept, {skipped_short} skipped (too short).")

# =========================
# Task 3: Overlay Templates and Save Plots
# =========================
print("\n=== Task 3: Overlaying Templates and Saving Plots/Arrays ===")
try:
    for det in DETECTORS:
        data = strain_data[det]
        time = data.times.value  # GWpy TimeSeries: .times.value gives numpy array

        print(f"Processing detector {det}...")
        for (m1, m2), template in templates.items():
            try:
                # Ensure template is a numpy array and matches data length
                template_np = template.numpy()
                if len(template_np) != len(data):
                    print(f"Template length mismatch for (m1={m1}, m2={m2}), skipping.")
                    continue

                # Plot overlay
                plt.figure(figsize=(10, 4))
                plt.plot(time, data.value, label=f"{det} Strain", color='black', linewidth=1)
                plt.plot(time, template_np, label=f"Template m1={m1}, m2={m2}", color='red', alpha=0.7)
                plt.xlabel("Time (s)")
                plt.ylabel("Strain")
                plt.title(f"{det} Strain with Template Overlay (m1={m1}, m2={m2})")
                plt.legend(loc='upper right')
                plt.tight_layout()

                # Save plot
                plot_filename = f"{det}_m1_{m1}_m2_{m2}.png"
                plot_path = os.path.join(PLOT_DIR, plot_filename)
                plt.savefig(plot_path)
                plt.close()

                # Save template array
                npy_filename = f"{det}_template_m1_{m1}_m2_{m2}.npy"
                npy_path = os.path.join(NPY_DIR, npy_filename)
                np.save(npy_path, template_np)

            except Exception as e:
                print(f"Error processing template (m1={m1}, m2={m2}) for {det}: {e}")

        print(f"Completed overlays for {det}.")

    print("All overlays and saves completed.")

except Exception as e:
    print(f"Critical error during overlay/save process: {e}")
    raise

print("\n=== Workflow Complete ===")
print(f"Plots saved in: {PLOT_DIR}")
print(f"Template arrays saved in: {NPY_DIR}")