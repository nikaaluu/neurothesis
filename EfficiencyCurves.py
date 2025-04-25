import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your semicolon-separated CSV
file_path = "/Users/nikalu/Downloads/Efficiency curves.csv"
df = pd.read_csv(file_path, sep=';')

# Confirm columns
print("Available columns:", df.columns.tolist())

# Pull log dilution and Ct values
if "Log₁₀(Dilution Factor)" in df.columns:
    log_dilution = df["Log₁₀(Dilution Factor)"].values
elif "Log10(Dilution Factor)" in df.columns:
    log_dilution = df["Log10(Dilution Factor)"].values
else:
    raise KeyError("Cannot find 'Log₁₀(Dilution Factor)' or 'Log10(Dilution Factor)' in columns!")

ct_tmr = df["Average Ct TMR"].values
ct_scg = df["Average Ct SCG"].values

# Function to calculate stats and plot
def regression_and_plot(ax, x, y, label, color):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept

    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    efficiency = (10**(-1 / slope) - 1) * 100

    ax.scatter(x, y, label=f'{label} data', edgecolors=color, facecolors='none', s=40)
    ax.plot(x, y_pred, color=color, label=f'{label} fit')
    ax.set_title(f"{label}", fontsize=12)
    ax.set_xlabel("Log₁₀(Dilution Factor)")
    ax.set_ylabel("Ct")
    ax.grid(True, linestyle=':')
    ax.legend(loc='lower left')
    ax.annotate(
        f"E = {efficiency:.1f}%\nR² = {r_squared:.3f}\nSlope = {slope:.2f}\nY-int = {intercept:.2f}",
        xy=(0.98, 0.98), xycoords='axes fraction',
        ha='right', va='top', fontsize=10, color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1)
    )

# Plot both graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

regression_and_plot(axes[0], log_dilution, ct_tmr, 'TMR', 'green')
regression_and_plot(axes[1], log_dilution, ct_scg, 'SCG', 'blue')

plt.suptitle("Primer Efficiency Curves", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
