#!/usr/bin/env python3
"""
GW150914 single-script matched filter + subtraction pipeline.

Requirements:
  - pycbc (with LALSuite)
  - gwpy
  - matplotlib
  - numpy
"""

import pylab
import numpy as np
from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass, matched_filter, sigma
from pycbc.waveform import get_td_waveform
from pycbc.psd import inverse_spectrum_truncation, interpolate

# ---------------- Step 0: Load GW150914 data ----------------
merger = Merger("GW150914")
strain = merger.strain('H1')

# ---------------- Step 1: Highpass + downsample ----------------
# Remove low frequency content and downsample to 2048 Hz
strain = resample_to_delta_t(highpass(strain, 15.0), 1.0/2048)

pylab.figure(figsize=[12,3])
pylab.plot(strain.sample_times, strain)
pylab.xlabel('Time (s)')
pylab.title('H1 Strain (raw)')
pylab.show()

# Remove 2 seconds from beginning and end to avoid boundary effects
conditioned = strain.crop(2,2)
pylab.figure(figsize=[12,3])
pylab.plot(conditioned.sample_times, conditioned)
pylab.xlabel('Time (s)')
pylab.title('H1 Strain (conditioned)')
pylab.show()

# ---------------- Step 2: PSD Estimation ----------------
# Welch method with 4-second segments
psd = conditioned.psd(4)
psd = interpolate(psd, conditioned.delta_f)
psd = inverse_spectrum_truncation(psd, int(4*conditioned.sample_rate),
                                  low_frequency_cutoff=15)

pylab.figure(figsize=[8,4])
pylab.loglog(psd.sample_frequencies, psd)
pylab.ylabel('$Strain^2/Hz$')
pylab.xlabel('Frequency (Hz)')
pylab.xlim(30, 1024)
pylab.title('PSD')
pylab.show()

# ---------------- Step 3: Generate Template ----------------
templates = {}  # store templates with (mass1, mass2) as key

for mass1 in range(20, 41):  # 20 to 40 M_sun
    for mass2 in range(20, 41):
        try:
            hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                     mass1=mass1,
                                     mass2=mass2,
                                     delta_t=conditioned.delta_t,
                                     f_lower=20)
            hp.resize(len(conditioned))
            # Shift waveform so merger occurs at first bin
            template = hp.cyclic_time_shift(hp.start_time)
            templates[(mass1, mass2)] = template
            print(f"Generated template for mass1={mass1}, mass2={mass2}")
        except Exception as e:
            print(f"Error generating template for mass1={mass1}, mass2={mass2}: {e}")
            templates[(mass1, mass2)] = None

pylab.figure(figsize=[12,3])
pylab.plot(template)
pylab.title('Template waveform')
pylab.show()
# ---------------- Step 4: Matched Filtering for all templates ----------------
best_snr = 0.0
best_mass = None
best_template = None
best_snr_series = None

for (mass1, mass2), template in templates.items():
    if template is None:
        continue
    # Matched filter
    snr = matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=20)
    # Crop edges affected by filtering
    snr = snr.crop(4 + 4, 4)  # 4s from PSD filter, 4s for template length
    
    # Find peak SNR
    peak_idx = abs(snr).numpy().argmax()
    max_snr = abs(snr[peak_idx])
    
    # Update best template if SNR higher
    if max_snr > best_snr:
        best_snr = max_snr
        best_mass = (mass1, mass2)
        best_template = template
        best_snr_series = snr

print(f"Best mass combination: m1={best_mass[0]} M_sun, m2={best_mass[1]} M_sun")
print(f"Maximum SNR: {best_snr:.3f}")

# Plot the SNR time series for the best template
import matplotlib.pyplot as plt
plt.figure(figsize=[10,4])
plt.plot(best_snr_series.sample_times, abs(best_snr_series))
plt.ylabel('SNR')
plt.xlabel('Time (s)')
plt.title(f'Matched Filter SNR for Best Template m1={best_mass[0]}, m2={best_mass[1]}')
plt.show()

# ---------------- Step 5: Align Best Template to SNR Peak ----------------
peak_idx = abs(best_snr_series).numpy().argmax()
snrp = best_snr_series[peak_idx]
peak_time = best_snr_series.sample_times[peak_idx]

dt = peak_time - conditioned.start_time
aligned = best_template.cyclic_time_shift(dt)
# Scale template to SNR=1
aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)
# Scale amplitude and phase to observed SNR peak
aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
aligned.start_time = conditioned.start_time

# ---------------- Step 6: Subtract Signal ----------------
subtracted = conditioned - aligned

# ---------------- Step 7: Visualize ----------------
for data, title in [(conditioned, 'Original H1 Data'),
                    (subtracted, 'Signal Subtracted from H1 Data')]:
    t, f, p = data.whiten(4,4).qtransform(.001,
                                          logfsteps=100,
                                          qrange=(8,8),
                                          frange=(20,512))
    pylab.figure(figsize=[15,3])
    pylab.pcolormesh(t, f, p**0.5, vmin=1, vmax=6)
    pylab.yscale('log')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Frequency (Hz)')
    pylab.title(title)
    pylab.xlim(merger.time-2, merger.time+1)
    pylab.show()
