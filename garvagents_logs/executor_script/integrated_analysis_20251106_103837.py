# --- Imports ---
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- Parameters and Setup ---
gps_center = 1180922494.5
duration = 64  # seconds
start = gps_center - duration / 2
end = gps_center + duration / 2
detectors = ['H1', 'L1']

low_freq = 35
high_freq = 350

event_gps = gps_center
zoom_window = 0.2  # seconds for merger plots and spectrograms
plot_start = event_gps - zoom_window
plot_end = event_gps + zoom_window

# Output directory
output_dir = "GW170608_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Task 1: Download and Validate Strain Data ---
print("\n=== Task 1: Downloading and validating strain data ===")
strain_data = {}

for det in detectors:
    try:
        print(f"Fetching {duration}s of strain data for {det} from {start} to {end}...")
        strain = TimeSeries.fetch_open_data(det, start, end, cache=True)
        print(f"Data fetched for {det}. Validating...")

        # Check for NaN or infinite values
        if np.isnan(strain.value).any():
            print(f"Validation failed: NaN values found in {det} data.")
        elif np.isinf(strain.value).any():
            print(f"Validation failed: Infinite values found in {det} data.")
        else:
            print(f"No NaN or infinite values in {det} data.")

        # Check data length
        expected_length = int(duration * strain.sample_rate.value)
        actual_length = len(strain)
        if actual_length == expected_length:
            print(f"Data length for {det} is correct: {actual_length} samples.")
        else:
            print(f"Warning: Data length for {det} is {actual_length}, expected {expected_length}.")

        # Check sampling rate
        print(f"Sampling rate for {det}: {strain.sample_rate.value} Hz")

        # Store strain data
        strain_data[det] = strain

        # Save raw data for reproducibility
        strain.write(os.path.join(output_dir, f"{det}_strain_raw.txt"), format='txt')
        print(f"Raw data saved for {det}.")

    except Exception as e:
        print(f"Error fetching or validating data for {det}: {e}")
        strain_data[det] = None

strain_H1 = strain_data['H1']
strain_L1 = strain_data['L1']

# --- Task 2: Bandpass Filtering ---
print("\n=== Task 2: Applying bandpass filter ({}-{} Hz) ===".format(low_freq, high_freq))
filtered_strain_H1 = None
filtered_strain_L1 = None

# Filtering for H1
try:
    if strain_H1 is not None:
        print("Applying bandpass filter (35-350 Hz) to H1 data...")
        filtered_strain_H1 = strain_H1.bandpass(low_freq, high_freq)
        if len(filtered_strain_H1) == len(strain_H1) and filtered_strain_H1.sample_rate == strain_H1.sample_rate:
            print("H1 filtering successful: Output length and sample rate match input.")
        else:
            print("Warning: H1 filtered data length or sample rate mismatch.")
        filtered_strain_H1.write(os.path.join(output_dir, "H1_strain_filtered.txt"), format='txt')
        print("Filtered H1 data saved.")
    else:
        print("No H1 strain data available for filtering.")
except Exception as e:
    print(f"Error filtering H1 data: {e}")

# Filtering for L1
try:
    if strain_L1 is not None:
        print("Applying bandpass filter (35-350 Hz) to L1 data...")
        filtered_strain_L1 = strain_L1.bandpass(low_freq, high_freq)
        if len(filtered_strain_L1) == len(strain_L1) and filtered_strain_L1.sample_rate == strain_L1.sample_rate:
            print("L1 filtering successful: Output length and sample rate match input.")
        else:
            print("Warning: L1 filtered data length or sample rate mismatch.")
        filtered_strain_L1.write(os.path.join(output_dir, "L1_strain_filtered.txt"), format='txt')
        print("Filtered L1 data saved.")
    else:
        print("No L1 strain data available for filtering.")
except Exception as e:
    print(f"Error filtering L1 data: {e}")

# --- Task 3: Time-domain Visualization ---
print("\n=== Task 3: Generating and saving time-domain plots ===")
output_files = {
    'H1': os.path.join(output_dir, 'GW170608_H1_filtered_strain_zoom.png'),
    'L1': os.path.join(output_dir, 'GW170608_L1_filtered_strain_zoom.png')
}
filtered_data = {
    'H1': filtered_strain_H1,
    'L1': filtered_strain_L1
}

for det, strain in filtered_data.items():
    try:
        if strain is None:
            print(f"Warning: No filtered strain data for {det}, skipping plot.")
            continue

        print(f"Extracting {det} strain data in [{plot_start}, {plot_end}] for plotting...")
        strain_zoom = strain.crop(plot_start, plot_end)
        times = strain_zoom.times.value - event_gps
        values = strain_zoom.value

        print(f"Plotting and saving time-domain strain for {det}...")
        plt.figure(figsize=(10, 4))
        plt.plot(times, values, label=f'{det} Filtered Strain', color='C0')
        plt.axvline(0, color='r', linestyle='--', label='Merger Time')
        plt.xlabel('Time (s) relative to GW170608')
        plt.ylabel('Strain')
        plt.title(f'{det} Filtered Strain around GW170608 (±0.2s)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_files[det])
        plt.close()
        print(f"Plot saved to {output_files[det]}")
    except Exception as e:
        print(f"Error generating plot for {det}: {e}")

# --- Task 4: Q-transform Spectrograms (with fallback) ---
print("\n=== Task 4: Generating and saving q-transform/spectrogram plots ===")
q_output_files = {
    'H1': os.path.join(output_dir, 'GW170608_H1_qtransform.png'),
    'L1': os.path.join(output_dir, 'GW170608_L1_qtransform.png')
}
spec_output_files = {
    'H1': os.path.join(output_dir, 'GW170608_H1_spectrogram.png'),
    'L1': os.path.join(output_dir, 'GW170608_L1_spectrogram.png')
}

for det, strain in filtered_data.items():
    try:
        if strain is None:
            print(f"Warning: No filtered strain data for {det}, skipping q-transform.")
            continue

        print(f"Cropping {det} strain data to [{plot_start}, {plot_end}] for q-transform...")
        strain_zoom = strain.crop(plot_start, plot_end)

        # Try q-transform
        try:
            print(f"Computing q-transform for {det}...")
            q = strain_zoom.q_transform(outseg=(plot_start, plot_end))
            print(f"Plotting q-transform for {det}...")
            fig = q.plot()
            ax = fig.gca()
            ax.axvline(event_gps, color='r', linestyle='--', label='Merger Time')
            ax.set_title(f"{det} Q-transform around GW170608 (±0.2s)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(q_output_files[det])
            plt.close(fig)
            print(f"Q-transform plot saved to {q_output_files[det]}")
        except Exception as qe:
            print(f"Q-transform failed for {det} ({qe}), falling back to standard spectrogram...")
            try:
                spec = strain_zoom.spectrogram(fftlength=0.02)
                fig = spec.plot()
                ax = fig.gca()
                ax.axvline(event_gps, color='r', linestyle='--', label='Merger Time')
                ax.set_title(f"{det} Spectrogram around GW170608 (±0.2s)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(spec_output_files[det])
                plt.close(fig)
                print(f"Spectrogram plot saved to {spec_output_files[det]}")
            except Exception as se:
                print(f"Spectrogram also failed for {det}: {se}")
    except Exception as e:
        print(f"Error processing {det}: {e}")

# --- Task 5: Structured Summary ---
print("\n=== Task 5: Structured summary of signal and detector comparison ===")
summary = {}

def estimate_signal_times(strain_zoom, threshold_factor=5):
    abs_strain = np.abs(strain_zoom.value)
    median = np.median(abs_strain)
    threshold = threshold_factor * median
    above = np.where(abs_strain > threshold)[0]
    if len(above) == 0:
        return None, None
    start_idx = above[0]
    end_idx = above[-1]
    times = strain_zoom.times.value
    return float(times[start_idx]), float(times[end_idx])

def estimate_freq_range_qtransform(q):
    try:
        data = q.data
        freqs = q.frequencies.value
        threshold = 0.1 * np.max(data)
        freq_mask = np.any(data > threshold, axis=1)
        if np.any(freq_mask):
            min_freq = float(freqs[freq_mask].min())
            max_freq = float(freqs[freq_mask].max())
            return min_freq, max_freq
    except Exception as e:
        print(f"Error estimating frequency range from q-transform: {e}")
    return None, None

def estimate_freq_range_spectrogram(spec):
    try:
        data = spec.data
        freqs = spec.frequencies.value
        threshold = 0.1 * np.max(data)
        freq_mask = np.any(data > threshold, axis=1)
        if np.any(freq_mask):
            min_freq = float(freqs[freq_mask].min())
            max_freq = float(freqs[freq_mask].max())
            return min_freq, max_freq
    except Exception as e:
        print(f"Error estimating frequency range from spectrogram: {e}")
    return None, None

for det, strain in filtered_data.items():
    det_summary = {}
    try:
        if strain is None:
            print(f"Warning: No filtered strain data for {det}, skipping summary.")
            summary[det] = {"error": "No data"}
            continue

        print(f"Cropping {det} strain for signal timing estimation...")
        strain_zoom = strain.crop(plot_start, plot_end)
        start_time, end_time = estimate_signal_times(strain_zoom)
        det_summary['signal_start_time'] = start_time
        det_summary['signal_end_time'] = end_time

        # Try q-transform for frequency range
        q = None
        min_freq = None
        max_freq = None
        try:
            print(f"Computing q-transform for {det} for frequency range estimation...")
            q = strain_zoom.q_transform(outseg=(plot_start, plot_end))
            min_freq, max_freq = estimate_freq_range_qtransform(q)
            det_summary['freq_range_method'] = 'q-transform'
        except Exception as qe:
            print(f"Q-transform failed for {det} ({qe}), falling back to spectrogram...")
            q = None

        if q is None or min_freq is None or max_freq is None:
            try:
                spec = strain_zoom.spectrogram(fftlength=0.02)
                min_freq, max_freq = estimate_freq_range_spectrogram(spec)
                det_summary['freq_range_method'] = 'spectrogram'
            except Exception as se:
                print(f"Spectrogram also failed for {det}: {se}")
                min_freq, max_freq = None, None
                det_summary['freq_range_method'] = 'none'

        det_summary['min_frequency_hz'] = min_freq
        det_summary['max_frequency_hz'] = max_freq

        snr_like = float(np.max(np.abs(strain_zoom.value)) / np.median(np.abs(strain_zoom.value)))
        det_summary['snr_like'] = snr_like

        summary[det] = det_summary

    except Exception as e:
        print(f"Error summarizing {det}: {e}")
        summary[det] = {"error": str(e)}

# Detector comparison
comparison = {}
try:
    if all(det in summary and 'snr_like' in summary[det] for det in ['H1', 'L1']):
        snr_diff = abs(summary['H1']['snr_like'] - summary['L1']['snr_like'])
        comparison['snr_like_difference'] = snr_diff
        comparison['stronger_signal'] = 'H1' if summary['H1']['snr_like'] > summary['L1']['snr_like'] else 'L1'
    else:
        comparison['snr_like_difference'] = None
        comparison['stronger_signal'] = None
except Exception as e:
    print(f"Error in detector comparison: {e}")
    comparison['error'] = str(e)

summary['comparison'] = comparison

# Print and save summary
print("\nStructured summary of signal analysis:")
print(json.dumps(summary, indent=2))

summary_path = os.path.join(output_dir, "GW170608_structured_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {summary_path}")

print("\n=== Analysis complete. All outputs saved in '{}' ===".format(output_dir))