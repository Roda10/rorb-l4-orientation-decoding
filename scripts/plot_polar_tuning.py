"""
plot_polar_tuning.py

Create polar tuning curves for VISp, VISal, and VISpm.

This gives a more neuroscience-style visualization of orientation/direction
selectivity across the 8 drifting grating angles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = "results/region_tuning_curves.csv"
FIGURES_DIR = "figures"

ORIENTATIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])


def plot_polar_tuning():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    region_summary = pd.read_csv(RESULTS_PATH)

    angles = np.deg2rad(ORIENTATIONS)

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for _, row in region_summary.iterrows():
        region = row["targeted_structure"]

        responses = np.array([row[f"mean_{ori}"] for ori in ORIENTATIONS])

        # close the circle
        angles_closed = np.concatenate([angles, [angles[0]]])
        responses_closed = np.concatenate([responses, [responses[0]]])

        ax.plot(
            angles_closed,
            responses_closed,
            marker="o",
            linewidth=2,
            label=region,
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{ori}°" for ori in ORIENTATIONS])

    ax.set_title("Polar tuning curves by cortical region", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    output_path = os.path.join(FIGURES_DIR, "polar_tuning_curves_by_region.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_polar_tuning()