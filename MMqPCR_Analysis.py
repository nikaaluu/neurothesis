import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

# === Step 1: Load Cq data file ===
file_path = "/Users/nikalu/Downloads/MMqPCR Analysis 03042025.csv"

# Auto-detect delimiter
with open(file_path, 'r', encoding='utf-8') as f:
    sample_lines = [next(f) for _ in range(5)]
delimiters = [',', ';', '\t', '|']
delimiter_counts = {d: sum(line.count(d) for line in sample_lines) for d in delimiters}
best_delim = max(delimiter_counts, key=delimiter_counts.get)

df = pd.read_csv(file_path, delimiter=best_delim)
df.columns = df.columns.str.strip()

# === Step 2: Rename columns for easier access ===
df = df.rename(columns={
    'Sample Name': 'Sample',
    'Cq_T': 'Cq_T',
    'Cq_S': 'Cq_S'
})

# === Step 3: Define PCR efficiencies ===
E_T = 0.986  # Telomere primer efficiency
E_S = 1.023  # SCG primer efficiency

# === Step 4: Efficiency-adjusted T/S calculation (inverted form) ===
df['T_S'] = ((E_S ** df['Cq_S']) / (E_T ** df['Cq_T'])) ** -1

# === Step 5: Normalize to the single 'Reference' sample ===
if 'Reference' not in df['Sample'].values:
    raise ValueError("No sample labeled exactly 'Reference' was found.")
ref_ts = df.loc[df['Sample'] == 'Reference', 'T_S'].values[0]
df['T_S_normalized'] = df['T_S'] / ref_ts

# === Step 6: Strip replicate IDs from non-reference samples (e.g., Sample_1 → Sample) ===
df.loc[df['Sample'] != 'Reference', 'Sample'] = (
    df.loc[df['Sample'] != 'Reference', 'Sample']
      .str.replace(r'[_\-]\d+$', '', regex=True)
)

# === Step 7: Group replicates (excluding Reference row) and compute summary statistics ===
df_no_ref = df[df['Sample'] != 'Reference'].copy()  # Explicit copy to avoid warnings
grouped = df_no_ref.groupby('Sample')
mean_df = grouped['T_S_normalized'].mean().reset_index(name='T_S_mean')
std_df = grouped['T_S_normalized'].std().reset_index(name='T_S_std')
summary_df = pd.merge(mean_df, std_df, on='Sample')

# Compute the Coefficient of Variation (CV) per sample
summary_df['CV'] = (summary_df['T_S_std'] / summary_df['T_S_mean']) * 100

# === Step 8: Compute z-score for each replicate in non-reference samples ===
df_no_ref['z_score'] = df_no_ref.groupby('Sample')['T_S_normalized'].transform(
    lambda x: (x - x.mean()) / x.std()
)
# Merge the z_score back into the main DataFrame for exporting
df['z_score'] = np.nan
df.loc[df_no_ref.index, 'z_score'] = df_no_ref['z_score']

# === NEW STEP: Merge CV per sample into main DataFrame ===
# Create a dictionary mapping sample name to CV from summary_df
cv_dict = summary_df.set_index('Sample')['CV'].to_dict()
# For non-reference rows, assign CV; for Reference, keep as NaN
df['CV'] = df['Sample'].apply(lambda s: cv_dict[s] if s in cv_dict else np.nan)

# === Step 9: Create plots ===
# Plot 1: Box-and-Whiskers plot of z-scores per sample
plt.figure(figsize=(10, 6))
unique_samples = df_no_ref['Sample'].unique()
data_to_plot = [df_no_ref.loc[df_no_ref['Sample'] == sample, 'z_score'] for sample in unique_samples]
plt.boxplot(data_to_plot, labels=unique_samples, patch_artist=True)
plt.xlabel('Sample')
plt.ylabel('Z-score')
plt.title('Distribution of Z-scores per Sample')
plt.xticks(rotation=45)
plt.tight_layout()
boxplot_fig = plt.gcf()  # Get current figure

# Plot 2: Bell curve (density plot) of all non-reference z-scores
plt.figure(figsize=(10, 6))
z_all = df_no_ref['z_score'].dropna()
count, bins, _ = plt.hist(z_all, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black')
mu, std_val = norm.fit(z_all)
xmin, xmax = plt.xlim()
x_vals = np.linspace(xmin, xmax, 100)
p = norm.pdf(x_vals, mu, std_val)
plt.plot(x_vals, p, 'r--', linewidth=2)
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.title('Bell Curve of Z-scores (All Non-reference Replicates)')
plt.tight_layout()
bellcurve_fig = plt.gcf()

# Plot 3: Bar plot of sample means with error bars (T_S_normalized) and CV
x = np.arange(len(summary_df))
colors = plt.cm.tab20(np.linspace(0, 1, len(summary_df)))
plt.figure(figsize=(12, 6))
bars = plt.bar(x, summary_df['T_S_mean'], yerr=summary_df['T_S_std'], capsize=5,
               edgecolor='black', color=colors)
plt.xticks(ticks=x, labels=summary_df['Sample'], rotation=45, fontsize=8)
plt.ylabel('Normalized T/S Ratio')
plt.title('Relative Telomere Length (MMqPCR, Inverted T/S Calculation)')
for bar, label in zip(bars, summary_df['Sample']):
    bar.set_label(label)
plt.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=9)
plt.tight_layout()
barplot_fig = plt.gcf()

# === Step 10: Create output folder with timestamp ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = os.path.join("/Users/nikalu/Downloads/MMqPCR Results Analysis", f"MMqPCR Results analysis {timestamp}")
os.makedirs(results_folder, exist_ok=True)

# Save plots
boxplot_path = os.path.join(results_folder, "boxplot_zscore.png")
bellcurve_path = os.path.join(results_folder, "bellcurve_zscore.png")
barplot_path = os.path.join(results_folder, "barplot_normalized_T_S.png")
boxplot_fig.savefig(boxplot_path)
bellcurve_fig.savefig(bellcurve_path)
barplot_fig.savefig(barplot_path)
plt.close('all')  # Close all figures

# Save CSV files
csv_all_path = os.path.join(results_folder, "mmqpcr_analysed_data.csv")
csv_summary_path = os.path.join(results_folder, "mmqpcr_analysed_data_summary.csv")
df.to_csv(csv_all_path, index=False)
summary_df.to_csv(csv_summary_path, index=False)

print("MMqPCR analysis complete. All plots and CSV files saved in:")
print(results_folder)
