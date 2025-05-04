# ======================================================
# Author: Nika Lu Yang
# Date: April 2025
# Bachelor Thesis Research (BTR): "Unraveling the Impact of Aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"
# Maastricht Science Programme (MSP)
# ======================================================

# === Import Required Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Step 1: Load the CSV File ===
# Load the semicolon-separated CSV file containing efficiency curve data.
file_path = "/Users/nikalu/Downloads/Efficiency curves.csv"
df = pd.read_csv(file_path, sep=';')

# Confirm the available columns in the CSV file.
print("Available columns:", df.columns.tolist())

# === Step 2: Extract Data from the CSV File ===
# Extract the log dilution factor and Ct values for TMR and SCG primers.
if "Log₁₀(Dilution Factor)" in df.columns:
    log_dilution = df["Log₁₀(Dilution Factor)"].values
elif "Log10(Dilution Factor)" in df.columns:
    log_dilution = df["Log10(Dilution Factor)"].values
else:
    raise KeyError("Cannot find 'Log₁₀(Dilution Factor)' or 'Log10(Dilution Factor)' in columns!")

# Extract Ct values for TMR and SCG primers.
ct_tmr = df["Average Ct TMR"].values
ct_scg = df["Average Ct SCG"].values

# === Step 3: Define the Regression and Plotting Function ===
def regression_and_plot(ax, x, y, label, color):
    """
    Perform linear regression, calculate efficiency, and plot the data.

    Parameters:
    - ax: The matplotlib axis to plot on.
    - x: The independent variable (Log₁₀(Dilution Factor)).
    - y: The dependent variable (Ct values).
    - label: The label for the data (e.g., 'TMR' or 'SCG').
    - color: The color for the plot.

    Outputs:
    - Scatter plot of the data points.
    - Regression line with annotations for efficiency, R², slope, and intercept.
    """
    # Calculate regression parameters
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept

    # Calculate R² and efficiency
    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    efficiency = (10**(-1 / slope) - 1) * 100

    # Plot the data and regression line
    ax.scatter(x, y, label=f'{label} data', edgecolors=color, facecolors='none', s=40)
    ax.plot(x, y_pred, color=color, label=f'{label} fit')
    ax.set_title(f"{label}", fontsize=12)
    ax.set_xlabel("Log₁₀(Dilution Factor)")
    ax.set_ylabel("Ct")
    ax.grid(True, linestyle=':')
    ax.legend(loc='lower left')

    # Annotate the plot with regression statistics
    ax.annotate(
        f"E = {efficiency:.1f}%\nR² = {r_squared:.3f}\nSlope = {slope:.2f}\nY-int = {intercept:.2f}",
        xy=(0.98, 0.98), xycoords='axes fraction',
        ha='right', va='top', fontsize=10, color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1)
    )

# === Step 4: Define the directory where the plot will be saved ===
save_path_main_directory = "/Users/nikalu/Downloads"
output_file = os.path.join(save_path_main_directory, "Primer Efficiency Curves.png")


# === Step 5: Plot Efficiency Curves ===
# Create a figure with two subplots for TMR and SCG primers.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
regression_and_plot(axes[0], log_dilution, ct_tmr, 'TMR', 'green')
regression_and_plot(axes[1], log_dilution, ct_scg, 'SCG', 'blue')
plt.suptitle("Primer Efficiency Curves", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95]) 

# Save the plot to the specified directory
plt.savefig(output_file, dpi=300)
print(f"Plot saved to: {output_file}")


