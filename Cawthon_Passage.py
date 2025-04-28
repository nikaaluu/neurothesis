import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

# === Step 1: Load Cq data file ===
file_path = "/Users/nikalu/Downloads/MMqPCR Analysis 03042025 No dilution.csv"

# Auto-detect delimiter
with open(file_path, 'r', encoding='utf-8') as f:
    sample_lines = [next(f) for _ in range(5)]
delimiters = [',', ';', '\t', '|']
delimiter_counts = {d: sum(line.count(d) for line in sample_lines) for d in delimiters}
best_delim = max(delimiter_counts, key=delimiter_counts.get)

df = pd.read_csv(file_path, delimiter=best_delim)
df.columns = df.columns.str.strip()

# === Step 2: Rename columns based on their position ===
new_names = list(df.columns)
if len(new_names) >= 4:
    new_names[1] = "Sample"
    new_names[2] = "Cq_T"
    new_names[3] = "Cq_S"
df.columns = new_names

# === Step 3: TMR, SCG, ΔΔCt, and T/S calculation ===
if 'Reference' not in df['Sample'].values:
    raise ValueError("No sample labeled exactly 'Reference' was found.")

# Reference Cq values
reference_row = df[df['Sample'] == 'Reference']
ref_telomere_cq = reference_row['Cq_T'].values[0]
ref_scg_cq = reference_row['Cq_S'].values[0]

# Calculate CqTMR and CqSCG
df['Cq_TMR'] = ref_telomere_cq - df['Cq_T'] 
df['Cq_SCG'] = ref_scg_cq - df['Cq_S'] 

# Calculate Delta Delta Cq
df['Delta_Delta_Cq'] = df['Cq_TMR'] - df['Cq_SCG']

# T/S = 2^(-ΔΔCq)
df['T_S'] = 2 ** (-df['Delta_Delta_Cq'])

# === Step 4: Clean replicate IDs ===
df.loc[df['Sample'] != 'Reference', 'Sample'] = (
    df.loc[df['Sample'] != 'Reference', 'Sample']
      .str.replace(r'[_\-]\d+$', '', regex=True)
)
# === Step 5: Add calculations (PLACEHOLDER1-4) ===
df['Total TL per diploid cell (kb)'] = df.apply(lambda row: f"{(1.23 * row['T_S'] * 1000):.2f} ± {(0.09 * row['T_S'] * 1000):.2f}", axis=1)
df['Average TL on each chromosome end (kb)'] = df.apply(lambda row: f"{(1.23 * row['T_S'] * 1000) / 92:.2f} ± {(0.09 * row['T_S'] * 1000) / 92:.2f}", axis=1)


# === Step 6: Group replicates (excluding Reference) and compute statistics ===
df_no_ref = df[df['Sample'] != 'Reference'].copy()
grouped = df_no_ref.groupby('Sample')
mean_df = grouped['T_S'].mean().reset_index(name='T_S_mean')
std_df = grouped['T_S'].std().reset_index(name='T_S_std')
summary_df = pd.merge(mean_df, std_df, on='Sample')

# Coefficient of Variation (CV)
summary_df['CV'] = (summary_df['T_S_std'] / summary_df['T_S_mean']) * 100

# === Step 7: Z-scores ===
df_no_ref['z_score'] = df_no_ref.groupby('Sample')['T_S'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Merge back z-scores
df['z_score'] = np.nan
df.loc[df_no_ref.index, 'z_score'] = df_no_ref['z_score']

# Merge CV into main DataFrame
cv_dict = summary_df.set_index('Sample')['CV'].to_dict()
df['CV'] = df['Sample'].apply(lambda s: cv_dict.get(s, np.nan))

# === Step 8: Create plots ===

# Plot 1: Boxplot
plt.figure(figsize=(10, 6))

samples = df_no_ref['Sample'].unique()
data_to_plot = [df_no_ref.loc[df_no_ref['Sample'] == sample, 'z_score'] for sample in samples]

plt.boxplot(data_to_plot, labels=samples, patch_artist=True)
plt.xlabel('Sample')
plt.ylabel('Z-score')
plt.title('Z-scores Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
boxplot_fig = plt.gcf()

# Plot 2: Bell curve
plt.figure(figsize=(10, 6))
z_all = df_no_ref['z_score'].dropna()
count, bins, _ = plt.hist(z_all, bins=10, density=True, alpha=0.6, edgecolor='black')
mu, std_val = norm.fit(z_all)
xmin, xmax = plt.xlim()
x_vals = np.linspace(xmin, xmax, 100)
p = norm.pdf(x_vals, mu, std_val)

plt.plot(x_vals, p, 'r--', linewidth=2)
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.title('Bell Curve')
plt.tight_layout()
bellcurve_fig = plt.gcf()

# Plot 3: Bar plot
x = np.arange(len(summary_df))
colors = plt.cm.tab20(np.linspace(0, 1, len(summary_df)))

plt.figure(figsize=(12, 6))
bars = plt.bar(x, summary_df['T_S_mean'], yerr=summary_df['T_S_std'], capsize=5,
               edgecolor='black', color=colors)
plt.xticks(ticks=x, labels=summary_df['Sample'], rotation=45)
plt.ylabel('T/S Ratio')
plt.title('Relative Telomere Length')
plt.legend(bars, summary_df['Sample'], title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
barplot_fig = plt.gcf()

# Plot 4: Grouped barplot by Passage
# Step 1: Extract Passage info
summary_df['Passage'] = summary_df['Sample'].str.extract(r'(P\d+)')

# Step 2: Find unique passages and samples
passages = sorted(summary_df['Passage'].dropna().unique(), key=lambda x: int(x[1:]))  # Sort by number
samples = summary_df['Sample'].unique()

# Step 3: Build data for plotting
bar_width = 0.1  # Slightly smaller bars if many samples
x = np.arange(len(passages))  # Base x-ticks for passages
offsets = np.linspace(-bar_width * len(samples) / 2, bar_width * len(samples) / 2, len(samples))

colors = plt.cm.tab20(np.linspace(0, 1, len(samples)))

# Step 4: Start plotting
plt.figure(figsize=(14, 6))

for i, sample in enumerate(samples):
    sample_data = summary_df[summary_df['Sample'] == sample]
    heights = []
    errors = []
    for passage in passages:
        row = sample_data[sample_data['Passage'] == passage]
        if not row.empty:
            heights.append(row['T_S_mean'].values[0])
            errors.append(row['T_S_std'].values[0])
        else:
            heights.append(np.nan)
            errors.append(np.nan)
    plt.bar(x + offsets[i], heights, width=bar_width, label=sample,
            yerr=errors, capsize=3, color=colors[i], edgecolor='black')

plt.xticks(ticks=x, labels=passages)
plt.xlabel('Passage', fontsize=12)
plt.ylabel('T/S Ratio', fontsize=12)
plt.title('Relative Telomere Length grouped by Passage', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Sample', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=9)
plt.tight_layout()

grouped_barplot_fig = plt.gcf()

# === Plot 5a: Total Telomere Length per diploid cell (averaged over replicates) ===

# Step 1: Calculate total and average telomere lengths directly from T_S_mean and T_S_std
summary_df['Total_mean'] = 1.23 * summary_df['T_S_mean'] * 1000
summary_df['Total_error'] = 0.09 * summary_df['T_S_mean'] * 1000
summary_df['Average_mean'] = summary_df['Total_mean'] / 92
summary_df['Average_error'] = summary_df['Total_error'] / 92

# Step 2: Prepare plotting
x = np.arange(len(summary_df))

# === Total Telomere Length Plot ===
plt.figure(figsize=(12, 6))
plt.bar(x, summary_df['Total_mean'], yerr=summary_df['Total_error'], capsize=5,
        color='skyblue', edgecolor='black', width=0.4)
plt.xticks(x, summary_df['Sample'], rotation=45)
plt.ylabel('Total TL (kb)')
plt.title('Total Telomere Length per Diploid Cell')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
total_telomere_fig = plt.gcf()

# === Plot 5b: Average Telomere Length per Chromosome End ===
plt.figure(figsize=(12, 6))
plt.bar(x, summary_df['Average_mean'], yerr=summary_df['Average_error'], capsize=5,
        color='lightgreen', edgecolor='black', width=0.4)
plt.xticks(x, summary_df['Sample'], rotation=45)
plt.ylabel('Average TL (kb)')
plt.title('Average Telomere Length per Chromosomal End')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
average_telomere_fig = plt.gcf()


# === Step 9: Save results ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = os.path.join("/Users/nikalu/Downloads/MMqPCR Results Analysis", f"MMqPCR_Results_{timestamp}")
os.makedirs(results_folder, exist_ok=True)

# Save plots
boxplot_fig.savefig(os.path.join(results_folder, "boxplot_zscore.png"))
bellcurve_fig.savefig(os.path.join(results_folder, "bellcurve_zscore.png"))
barplot_fig.savefig(os.path.join(results_folder, "barplot_TS_individual.png"))
grouped_barplot_fig.savefig(os.path.join(results_folder, "barplot_TS_grouped_by_passage.png"))
total_telomere_fig.savefig(os.path.join(results_folder, "total_telomere_length_plot.png"))
average_telomere_fig.savefig(os.path.join(results_folder, "average_telomere_length_plot.png"))
plt.close('all')

# Save CSVs
df.to_csv(os.path.join(results_folder, "mmqpcr_analysed_data.csv"), index=False)
summary_df.to_csv(os.path.join(results_folder, "mmqpcr_analysed_data_summary.csv"), index=False)

print("MMqPCR analysis complete. Results saved in:")
print(results_folder)
