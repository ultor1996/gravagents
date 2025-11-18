# ============================================================
# GW150914 Strain Data Download and Q-transform Spectroscopy
# ============================================================

import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# -------------------------------
# 1. Define Event Time and Window
# -------------------------------
gw150914_gps = 1126259462.4  # GW150914 event GPS time
duration = 4  # seconds of data to analyze
start = gw150914_gps - 2
end = gw150914_gps + 2

# -----------------------------------
# 2. Download Strain Data for H1 & L1
# -----------------------------------
h1_strain = None
l1_strain = None

try:
    print("[INFO] Fetching H1 strain data...")
    h1_strain = TimeSeries.fetch_open_data('H1', start, end, cache=True)
    print("[SUCCESS] H1 strain data fetched successfully.")
except Exception as e:
    print(f"[ERROR] Error fetching H1 data: {e}")

try:
    print("[INFO] Fetching L1 strain data...")
    l1_strain = TimeSeries.fetch_open_data('L1', start, end, cache=True)
    print("[SUCCESS] L1 strain data fetched successfully.")
except Exception as e:
    print(f"[ERROR] Error fetching L1 data: {e}")

# -----------------------------------
# 3. Q-transform Spectroscopy Plots
# -----------------------------------
def plot_q_transform(strain, detector_label, event_gps):
    """
    Plot the Q-transform for a given strain time series.
    """
    try:
        print(f"[INFO] Generating Q-transform plot for {detector_label}...")
        q = strain.q_transform(outseg=(event_gps - 0.5, event_gps + 0.5))
        fig = q.plot(
            norm='log',
            vmin=1e-24,
            vmax=1e-21,
            cmap='viridis'
        )
        ax = fig.gca()
        ax.set_title(f"GW150914 {detector_label} Q-transform")
        # Add a normalized energy colorbar
        cbar = fig.colorbar(label="Normalized energy")
        plt.show()
        print(f"[SUCCESS] {detector_label} Q-transform plot generated.")
    except Exception as e:
        print(f"[ERROR] Error plotting {detector_label} Q-transform: {e}")

# Plot for H1
if h1_strain is not None:
    plot_q_transform(h1_strain, "H1", gw150914_gps)
else:
    print("[WARN] H1 strain data not available. Skipping H1 Q-transform plot.")

# Plot for L1
if l1_strain is not None:
    plot_q_transform(l1_strain, "L1", gw150914_gps)
else:
    print("[WARN] L1 strain data not available. Skipping L1 Q-transform plot.")

# -----------------------------------
# 4. Save Results for Further Use
# -----------------------------------
gw150914_h1_strain = h1_strain
gw150914_l1_strain = l1_strain

print("[INFO] Script execution complete. Strain data variables are available for further analysis.")