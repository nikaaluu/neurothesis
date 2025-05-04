# ======================================================
# Author: Nika Lu Yang
# Date: April 2025
# Bachelor Thesis Research (BTR): "Unraveling the Impact of Aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"
# Maastricht Science Programme (MSP)
# ======================================================

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

# === Step 1: Load CSV file ===
file_path_csv = "/Users/nikalu/Downloads/dsDNA 4_30_2025 4_19_01 PM.csv"
with open(file_path_csv, 'r') as f:
    csv_lines = f.readlines()

# === Step 2: Parse Samples from the CSV File ===
# The CSV file contains multiple samples, each starting with "Sample:".
samples = []  # List to store raw data for each sample
sample_names = []  # List to store sample names
current_sample = []  # Temporary storage for the current sample

for line in csv_lines:
    stripped = line.strip()
    if stripped.startswith("Sample:"):
        if current_sample:
            samples.append(current_sample)
        current_sample = [stripped]
        name = stripped.split(":", 1)[1].strip()  # Extract sample name
        sample_names.append(name)
    else:
        current_sample.append(stripped)

# Add last sample to the list
if current_sample:
    samples.append(current_sample)

# === Step 3: Extract Absorbance Data for Each Sample ===
# Parse spectral data (wavelength and absorbance) from each sample
parsed_samples = []
for sample in samples:
    spectral_data = []
    for line in sample[14:]:  # Spectral data typically starts at line 14
        parts = line.strip().split(',')
        if len(parts) == 2:
            try:
                wl = float(parts[0]) # Wavelength in nm
                abs_val = float(parts[1]) # Absorbance value
                spectral_data.append((wl, abs_val))
            except ValueError:
                continue
    if spectral_data:
        # Create DataFrame for the spectral data
        df = pd.DataFrame(spectral_data, columns=["Wavelength (nm)", "Absorbance"])
        df = df.drop_duplicates(subset="Wavelength (nm)").sort_values("Wavelength (nm)")
        parsed_samples.append(df)

# === Step 4: Plot All Samples with Nanodrop Style === 
plt.figure(figsize=(12, 6))
# Distinct colors for each sample
distinct_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
    '#bcbd22', '#17becf'
]

# Plot each sample with a distinct color
for i, df in enumerate(parsed_samples):
    x = df["Wavelength (nm)"].values
    y = df["Absorbance"].values
    x_smooth = np.linspace(x.min(), x.max(), 500)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    color = distinct_colors[i % len(distinct_colors)]
    plt.fill_between(x_smooth, y_smooth, alpha=0.2, color=color)
    plt.plot(x_smooth, y_smooth, color=color, label=sample_names[i])

# === Step 5: Define the directory where the plot will be saved ===
save_path_main_directory = "/Users/nikalu/Downloads"
output_file = os.path.join(save_path_main_directory, "dsDNA_absorbance_plot.png")

# === Step 6: Final Styling ===
plt.title("dsDNA Absorbance", fontsize=14, color='black')
plt.xlabel("Wavelength (nm)")
plt.ylabel("10 mm Absorbance")
plt.grid(False)
plt.xlim(220, 350)  # Adjust x-axis limit as needed
plt.ylim(0, 2.5)  # Adjust y-axis limit as needed
plt.xticks()
plt.yticks()
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot to the specified directory
plt.savefig(output_file, dpi=300)
print(f"Plot saved to: {output_file}")




