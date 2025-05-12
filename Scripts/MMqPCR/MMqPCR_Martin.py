import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm
from pathlib import Path
import datetime
import seaborn as sns
import re

# === Utility Functions ===
def detect_delimiter(file_path, n_lines=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample_lines = [next(f) for _ in range(n_lines)]
    delimiters = [',', ';', '\t', '|']
    return max(delimiters, key=lambda d: sum(line.count(d) for line in sample_lines))

def plot_bar(data, x_col, y_col, yerr_col, title, save_path, xlabel='Sample'):
    x = np.arange(len(data))
    colors = plt.cm.tab20(np.linspace(0, 1, len(data)))
    plt.figure(figsize=(10, 6))
    plt.bar(x, data[y_col], yerr=data[yerr_col], capsize=5, edgecolor='black', color=colors)
    plt.xticks(ticks=x, labels=data[x_col], rotation=45, fontsize=8)
    plt.ylabel('T/S Ratio')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === Load and Preprocess Data ===
file_path = Path("/Users/nikalu/Downloads/MMqPCR_Real2.csv")
delimiter = detect_delimiter(file_path)
df = pd.read_csv(file_path, delimiter=delimiter)
df.columns = df.columns.str.strip()

# Rename key columns
if len(df.columns) >= 4:
    df.columns = [df.columns[0], "Sample", "Cq_T", "Cq_S"] + list(df.columns[4:])

# PCR efficiencies
E_T, E_S = 1.059, 0.992

# Calculate T/S ratio
df['T_S'] = ((E_T ** df['Cq_T']) / (E_S ** df['Cq_S'])) ** -1
df['Sample'] = df['Sample'].str.replace(r'[_\-]\d+$', '', regex=True).str.strip()

# Extract passage and cell line info early
df['Passage'] = df['Sample'].str.extract(r'P(\d+)', expand=False).fillna('Unknown')
df['Cell_Line'] = df['Sample'].str.extract(r'^(.*?)(?: P\d+|$)', expand=False).str.strip()

# Filter for replicates
df_no_ref = df[~df['Sample'].isin(['Reference', 'NTR'])].copy()

# Summary statistics
summary_df = (
    df_no_ref.groupby('Sample')['T_S']
    .agg(['mean', 'std'])
    .rename(columns={'mean': 'T_S_mean', 'std': 'T_S_std'})
    .reset_index()
)
summary_df['CV'] = 100 * summary_df['T_S_std'] / summary_df['T_S_mean']
df = df.merge(summary_df[['Sample', 'CV']], on='Sample', how='left')

# Z-scores
df_no_ref['z_score'] = df_no_ref.groupby('Sample')['T_S'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['z_score'] = np.nan
df.loc[df_no_ref.index, 'z_score'] = df_no_ref['z_score']

# === Output folder ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = Path("/Users/nikalu/Downloads/MMqPCR Results Analysis") / f"MMqPCR Results analysis {timestamp}"
results_folder.mkdir(parents=True, exist_ok=True)

# === Plots ===
# Boxplot z-scores
plt.figure(figsize=(10, 6))
df_no_ref.boxplot(column='z_score', by='Sample', grid=False)
plt.title('Z-score Distribution')
plt.suptitle("")
plt.xlabel('Sample')
plt.ylabel('Z-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(results_folder / "boxplot_zscore.png")
plt.close()

# Bell curve
z_all = df_no_ref['z_score'].dropna()
mu, std_val = norm.fit(z_all)
x_vals = np.linspace(min(z_all), max(z_all), 100)
plt.figure(figsize=(10, 6))
plt.hist(z_all, bins=10, density=True, alpha=0.6, edgecolor='black')
plt.plot(x_vals, norm.pdf(x_vals, mu, std_val), 'r--', linewidth=2)
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.title('Bell Curve of Z-scores')
plt.tight_layout()
plt.savefig(results_folder / "bellcurve_zscore.png")
plt.close()

# Barplot of summary (unordered)
plot_bar(summary_df, 'Sample', 'T_S_mean', 'T_S_std', 'Telomere Length (T/S Ratio)', results_folder / "barplot_T_S_unordered.png")

# Grouped and Ordered Bar Plot: Cell Line and Passage Ordered, with Spacing

# Define desired order
cell_line_order = ["hNESC 338", "hNESC A13", "hNESC T4.13 GC", "hNESC T4.13 M", "iPSC T4.13 GC", "iPSC T4.13 M", "MG63"]
passage_order = ["5", "6", "7", "18", "19", "20"]

# Create sort key
def match_sample(row):
    for cl in cell_line_order:
        if cl in row['Sample']:
            for p in passage_order:
                if f"P{p}" in row['Sample']:
                    return (cell_line_order.index(cl), int(p))
    return (len(cell_line_order), 99)

# Ensure info is available
summary_df = summary_df.merge(df[['Sample', 'Cell_Line', 'Passage']].drop_duplicates(), on='Sample', how='left')
summary_df['sort_key'] = summary_df.apply(match_sample, axis=1)
summary_df = summary_df.sort_values('sort_key')

# Insert spacing rows
display_rows = []
last_line = None
for _, row in summary_df.iterrows():
    if row['Cell_Line'] != last_line and last_line is not None:
        display_rows.append({'Sample': '', 'T_S_mean': np.nan, 'T_S_std': np.nan, 'Cell_Line': '', 'Passage': '', 'sort_key': ('', 0)})
    display_rows.append(row.to_dict())
    last_line = row['Cell_Line']

display_df = pd.DataFrame(display_rows)
display_df['Passage'] = display_df['Passage'].astype(str)

# Filter out rows with unknown cell lines or passages
display_df = display_df[
    display_df['Cell_Line'].isin(cell_line_order) &
    display_df['Passage'].astype(str).isin(passage_order) &
    display_df['Sample'].notna() & (display_df['Sample'] != '')
]

# Plot
x = np.arange(len(display_df))
color_map = {
    "hNESC 338": 'steelblue',
    "hNESC A13": 'crimson',
    "hNESC T4.13 GC": 'orchid',
    "hNESC T4.13 M": 'olive',
    #"iPSC T4.13 GC": 'cyan',
    #"iPSC T4.13 M": 'skyblue',
    #"MG63": 'gray'
}
colors = [color_map.get(cl, 'gray') for cl in display_df['Cell_Line']]

plt.figure(figsize=(14, 6))
bars = plt.bar(x, display_df['T_S_mean'], yerr=display_df['T_S_std'], capsize=5, edgecolor='black', color=colors)

# Only show passage number on x-axis
labels = display_df['Passage'].replace('', ' ')
plt.xticks(ticks=x, labels=labels, rotation=45, ha='right', fontsize=8)
plt.ylabel('T/S Ratio')
plt.title('Telomere Length (T/S Ratio) by Sample (Grouped by Cell Line and Passage)')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_map[cl], label=cl) for cl in color_map]
plt.legend(handles=legend_elements, title='Cell Line', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig(results_folder / "barplot_T_S_grouped.png")
plt.close()

# === Grouping ===
df['Passage'] = df['Sample'].str.extract(r'P(\d+)', expand=False).fillna('Unknown')
df['Cell_Line'] = df['Sample'].str.extract(r'^(.*?)(?: P\d+|$)', expand=False).str.strip()
df['Age_Group'] = df['Passage'].apply(lambda p: 'Young' if p in ['5', '6', '7'] else ('Old' if p in ['18', '19', '20'] else 'Unknown'))
df['Age_Cell_Group'] = df.apply(lambda r: f"{r['Age_Group']} {r['Cell_Line']}" if r['Age_Group'] != 'Unknown' else 'Unknown', axis=1)
df_no_ref = df[~df['Sample'].isin(['Reference', 'NTR'])].copy()

# Barplot young vs old by Cell Line
grouped_age_cell = (
    df_no_ref[df_no_ref['Age_Cell_Group'] != 'Unknown']
    .groupby(['Cell_Line', 'Age_Group'])['T_S']
    .agg(['mean', 'std'])
    .reset_index()
)
grouped_age_cell['Combined_Label'] = grouped_age_cell['Age_Group'] + " " + grouped_age_cell['Cell_Line']
plot_bar(grouped_age_cell, 'Combined_Label', 'mean', 'std', 'Telomere Length by Age Group', results_folder / "barplot_young_vs_old_by_cell_line.png")

# Side-by-side barplot
pivoted = grouped_age_cell.pivot(index='Cell_Line', columns='Age_Group', values='mean')
pivoted_std = grouped_age_cell.pivot(index='Cell_Line', columns='Age_Group', values='std')
x = np.arange(len(pivoted))
bar_width = 0.3
plt.figure(figsize=(6, 4))
# Plot bars
bars1 = plt.bar(x - bar_width/2, pivoted['Young'], label='Young', color='skyblue', edgecolor='black', width=bar_width)
bars2 = plt.bar(x + bar_width/2, pivoted['Old'], label='Old', color='salmon', edgecolor='black', width=bar_width)

# Add error bars with caps and horizontal lines (whiskers)
plt.errorbar(
    x - bar_width/2,
    pivoted['Young'],
    yerr=pivoted_std['Young'],
    fmt='none',
    ecolor='black',
    capsize=5,
    elinewidth=1,
    capthick=1
)
plt.errorbar(
    x + bar_width/2,
    pivoted['Old'],
    yerr=pivoted_std['Old'],
    fmt='none',
    ecolor='black',
    capsize=5,
    elinewidth=1,
    capthick=1
)

plt.ylabel('T/S Ratio')
plt.title('Telomere Length')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(results_folder / "barplot_young_vs_old_side_by_side.png")
plt.close()

# === T-tests and ANOVA ===
cell_lines = ["hNESC 338", "hNESC A13", "hNESC T4.13 GC", "hNESC T4.13 M"]
young_passages, old_passages = ["5", "6", "7"], ["18", "19", "20"]

print("\nT-test results (Young vs Old):")
for cl in cell_lines:
    y = df_no_ref[(df_no_ref['Cell_Line'] == cl) & df_no_ref['Passage'].isin(young_passages)]['T_S']
    o = df_no_ref[(df_no_ref['Cell_Line'] == cl) & df_no_ref['Passage'].isin(old_passages)]['T_S']
    if len(y) > 1 and len(o) > 1:
        t_stat, p_val = ttest_ind(y, o, equal_var=False)
        print(f"{cl}: t = {t_stat:.3f}, p = {p_val:.4f} (n_young={len(y)}, n_old={len(o)})")
    else:
        print(f"{cl}: Not enough data for t-test")

anova_df = df_no_ref[df_no_ref['Cell_Line'].isin(cell_lines) & df_no_ref['Age_Group'].isin(['Young', 'Old'])].copy()
model = ols('T_S ~ C(Cell_Line) + C(Age_Group) + C(Cell_Line):C(Age_Group)', data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nTwo-way ANOVA results:")
print(anova_table.round(4))

# === Save summary data ===
# Collect t-test results in a list
t_test_results = []

print("\nT-test results (Young vs Old):")
for cl in cell_lines:
    y = df_no_ref[(df_no_ref['Cell_Line'] == cl) & df_no_ref['Passage'].isin(young_passages)]['T_S']
    o = df_no_ref[(df_no_ref['Cell_Line'] == cl) & df_no_ref['Passage'].isin(old_passages)]['T_S']
    if len(y) > 1 and len(o) > 1:
        t_stat, p_val = ttest_ind(y, o, equal_var=False)
        result = {
            "Cell_Line": cl,
            "t_statistic": t_stat,
            "p_value": p_val,
            "n_young": len(y),
            "n_old": len(o),
            "Significant": 'Yes' if p_val < 0.05 else 'No'
        }
    else:
        result = {
            "Cell_Line": cl,
            "t_statistic": None,
            "p_value": None,
            "n_young": len(y),
            "n_old": len(o),
            "Significant": "Not enough data"
        }
    t_test_results.append(result)

# Save t-test results
t_test_df = pd.DataFrame(t_test_results)
t_test_df.to_csv(results_folder / "t_test_results.csv", index=False)


anova_table.to_csv(results_folder / "anova_results.csv")


# === Save all Data ===
df.to_csv(results_folder / "mmqpcr_analysed_data_with_groups.csv", index=False)
summary_df.to_csv(results_folder / "mmqpcr_analysed_data_summary.csv", index=False)
print("\nAll analysis complete. Outputs saved to:")
print(results_folder)

# === Barplot Ordered by Passage ===
# Re-define sort key: group first by passage, then by cell line
def match_by_passage(row):
    passage_idx = passage_order.index(row['Passage']) if row['Passage'] in passage_order else 99
    cell_idx = cell_line_order.index(row['Cell_Line']) if row['Cell_Line'] in cell_line_order else 99
    return (passage_idx, cell_idx)

summary_df['sort_key_passage'] = summary_df.apply(match_by_passage, axis=1)
summary_by_passage = summary_df.sort_values('sort_key_passage')

# Plot: Grouped by Passage
x = np.arange(len(summary_by_passage))
colors = [color_map.get(cl, 'gray') for cl in summary_by_passage['Cell_Line']]

plt.figure(figsize=(14, 6))
plt.bar(
    x,
    summary_by_passage['T_S_mean'],
    yerr=summary_by_passage['T_S_std'],
    capsize=5,
    edgecolor='black',
    color=colors
)
labels = summary_by_passage['Sample']
plt.xticks(ticks=x, labels=labels, rotation=45, ha='right', fontsize=8)
plt.ylabel('T/S Ratio')
plt.title('Telomere Length (T/S Ratio)')

# Add legend
legend_elements = [Patch(facecolor=color_map[cl], label=cl) for cl in color_map]
plt.legend(handles=legend_elements, title='Cell Line', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig(results_folder / "barplot_T_S_grouped_by_passage.png")
plt.close()

# === Heatmap of T/S Ratios ===
# Filter and pivot data
heatmap_data = df_no_ref[df_no_ref['Cell_Line'].isin(cell_line_order) & df_no_ref['Passage'].isin(passage_order)]

# Aggregate mean T/S per cell line and passage
heatmap_matrix = (
    heatmap_data
    .groupby(['Cell_Line', 'Passage'])['T_S']
    .mean()
    .unstack()
    .reindex(index=cell_line_order, columns=passage_order)
)

# Heatmap Plot
"""
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': 'Mean T/S Ratio'}
)
plt.title("Heatmap of Mean Telomere Length (T/S Ratio)")
plt.xlabel("Passage")
plt.ylabel("Cell Line")
plt.tight_layout()
plt.savefig(results_folder / "heatmap_T_S_ratio.png")
plt.close()
"""
# === Filter cell lines with valid passages only ===
valid_passages = set(passage_order)

# Identify cell lines that have at least one T/S value for a valid passage
valid_cell_lines = (
    df_no_ref[df_no_ref['Passage'].isin(valid_passages)]
    .groupby('Cell_Line')['T_S']
    .count()
    .loc[lambda x: x > 0]
    .index
)

# Now create the heatmap data using only valid cell lines
filtered_heatmap_data = df_no_ref[
    df_no_ref['Cell_Line'].isin(valid_cell_lines) &
    df_no_ref['Passage'].isin(valid_passages)
]

heatmap_matrix_filtered = (
    filtered_heatmap_data
    .groupby(['Cell_Line', 'Passage'])['T_S']
    .mean()
    .unstack()
    .reindex(index=[cl for cl in cell_line_order if cl in valid_cell_lines], columns=passage_order)
)

# Plot filtered heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_matrix_filtered,
    annot=False,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': 'Mean T/S Ratio'}
)
plt.title("Mean T/S Ratio")
plt.xlabel("Passage")
plt.ylabel("Cell Line")
plt.tight_layout()
plt.savefig(results_folder / "heatmap_T_S_ratio_filtered.png")
plt.close()
