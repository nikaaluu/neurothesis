# ======================================================
# Author: Nika Lu Yang
# Date: April 2025
# Bachelor Thesis Research (BTR): "Unraveling the Impact of Aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"
# Maastricht Science Programme (MSP)
# ======================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import f_oneway, levene, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from pingouin import welch_anova, pairwise_gameshowell
from scikit_posthocs import posthoc_dunn
import seaborn as sns
import matplotlib.gridspec as gridspec

# === Step 1: Define input and output directories ===
input_csv = "summary_per_well.csv"
analysis_output_root = "53BP1_Stats_Results"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
script_name = os.path.splitext(os.path.basename(__file__))[0]
run_folder = os.path.join(analysis_output_root, f'Analysis_{timestamp}_{script_name}')
os.makedirs(run_folder, exist_ok=True)

# === Step 2: Load and prepare data ===
df = pd.read_csv(input_csv)
df.rename(columns={
    'passage': 'Passage',
    'cell_line': 'Line',
    'foci_per_cell': 'Foci_per_Cell'
}, inplace=True)
df['Passage'] = df['Passage'].astype(str)
df['Line']    = df['Line'].astype(str)

# === Step 3: Determine numeric sort order for passages and lines ===
passage_order = sorted(df['Passage'].unique(), key=lambda p: int(re.search(r'\d+', p).group()))
line_order    = sorted(df['Line'].unique())

summary_rows    = []
all_combined_df = []

# === Step 4: Plot per-passage plots and summaries from youngest to oldest ===
for passage in passage_order:
    sub = df[df['Passage'] == passage]
    passage_folder = os.path.join(run_folder, passage)
    os.makedirs(passage_folder, exist_ok=True)

    data_per_line = [sub[sub['Line'] == ln]['Foci_per_Cell'] for ln in line_order]

    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data_per_line,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor='lightblue', color='blue', linewidth=1.5),
        medianprops=dict(color='red', linewidth=2),
        meanprops=dict(marker='o', markerfacecolor='green', markersize=8),
        whiskerprops=dict(color='blue', linewidth=1.5),
        capprops=dict(color='blue', linewidth=1.5),
        flierprops=dict(marker='x', color='gray', markersize=6)
    )
    plt.xticks(np.arange(1, len(line_order) + 1), labels=line_order, rotation=45)
    plt.title(f"{passage} – Foci per Cell (Boxplot)")
    plt.xlabel("Line")
    plt.ylabel("Foci per Cell")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(passage_folder, f"{passage}_Boxplot.png"), dpi=300)
    plt.close()

    # Violin plot
    plt.figure(figsize=(8, 6))
    plt.violinplot(data_per_line, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(line_order) + 1), labels=line_order)
    plt.title(f"{passage} – Foci per Cell (Violin)")
    plt.xlabel("Line")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()
    plt.savefig(os.path.join(passage_folder, f"{passage}_Violin.png"))
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 6))
    for ln in line_order:
        plt.hist(sub[sub['Line'] == ln]['Foci_per_Cell'], bins=20,
                 alpha=0.5, label=ln, edgecolor='black')
    plt.title(f"{passage} – Foci per Cell (Histogram)")
    plt.xlabel("Foci per Cell")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(passage_folder, f"{passage}_Hist.png"))
    plt.close()

    all_combined_df.append(sub)

    # Summary statistics
    for ln in line_order:
        data = sub[sub['Line'] == ln]['Foci_per_Cell']
        summary_rows.append({
            'Passage': passage,
            'Line': ln,
            'N': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        })

# Save per-passage summary table
pd.DataFrame(summary_rows)\
  .to_csv(os.path.join(run_folder, "Summary_Statistics_Per_Passage.csv"), index=False)

# === Step 5: Plot all passages and all lines (global) ===
all_df = pd.concat(all_combined_df, ignore_index=True)
all_df['Group'] = all_df['Passage'] + " - " + all_df['Line']

# Sort groups by numeric passage then line
groups = sorted(all_df['Group'].unique(),
                key=lambda g: (int(re.search(r'P(\d+)', g).group(1)),
                               line_order.index(g.split(" - ")[1])))
data_per_group = [all_df[all_df['Group'] == g]['Foci_per_Cell'] for g in groups]

# Global Boxplot
plt.figure(figsize=(max(10, len(groups)), 6))
plt.boxplot(data_per_group, patch_artist=True, showmeans=True)
plt.xticks(np.arange(1, len(groups) + 1), labels=groups, rotation=90)
plt.title("Global Foci per Cell (Boxplot)")
plt.xlabel("Passage - Line")
plt.ylabel("Foci per Cell")
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Combined_Boxplot.png"), dpi=300)
plt.close()

# === Step 6: Statistics ===
# 1) Passages within each Line
anova_passage = []
tukey_passage = []
for ln, grp in all_df.groupby('Line'):
    data = [grp[grp['Passage']==p]['Foci_per_Cell'] for p in passage_order if p in grp['Passage'].unique()]
    if len(data) > 1:
        stat, pval = f_oneway(*data)
        anova_passage.append({'Line': ln, 'ANOVA_p': pval})
        tk = pairwise_tukeyhsd(grp['Foci_per_Cell'], grp['Passage'], alpha=0.05)
        for row in tk.summary().data[1:]:
            tukey_passage.append({
                'Line': ln,
                'Comparison': f"{row[0]} vs {row[1]}",
                'p-value': row[5],
                'Significant': 'Yes' if row[6] else 'No'
            })

pd.DataFrame(anova_passage)\
  .to_csv(os.path.join(run_folder, "ANOVA_Passages_within_Lines.csv"), index=False)
pd.DataFrame(tukey_passage)\
  .to_csv(os.path.join(run_folder, "Tukey_Passages_within_Lines.csv"), index=False)

# 2) Lines within each Passage
anova_line = []
for passage in passage_order:
    grp = all_df[all_df['Passage']==passage]
    data = [grp[grp['Line']==ln]['Foci_per_Cell'] for ln in line_order if ln in grp['Line'].unique()]
    if len(data) > 1:
        stat, pval = f_oneway(*data)
        anova_line.append({'Passage': passage, 'ANOVA_p': pval})

pd.DataFrame(anova_line)\
  .to_csv(os.path.join(run_folder, "ANOVA_Lines_within_Passages.csv"), index=False)

# 3) Two-way ANOVA
model = ols("Foci_per_Cell ~ C(Passage) * C(Line)", data=all_df).fit()
anova2 = anova_lm(model)
anova2.to_csv(os.path.join(run_folder, "Two_Way_ANOVA.csv"))

# 4) Mixed Linear Model
mlm = smf.mixedlm("Foci_per_Cell ~ Passage", all_df, groups=all_df["Line"]).fit()
with open(os.path.join(run_folder, "MixedLM.txt"), "w") as f:
    f.write(mlm.summary().as_text())

# 5) Density Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data=all_df, x='Foci_per_Cell', hue='Passage', fill=True,
            hue_order=passage_order)
plt.title("Density of Foci per Cell Across Passages")
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Density_Plot.png"))
plt.close()

# 6) Heatmap of mean Foci_per_Cell, sorted bottom-to-top
pivot = all_df.pivot_table(
    values='Foci_per_Cell', index='Passage', columns='Line', aggfunc='mean'
)
pivot = pivot.reindex(index=passage_order, columns=line_order)

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(1,2, width_ratios=[20,1], wspace=0.05)
ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
sns.heatmap(pivot, ax=ax, cbar_ax=cax, cmap='coolwarm', linewidths=0.5,
            cbar_kws={'label': 'Mean Foci per Cell'})
ax.set_title("Mean Foci per Cell")
ax.set_ylabel("Passage")
ax.set_xlabel("Line")
ax.invert_yaxis()  # flip so lowest passage appears at bottom
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Heatmap.png"), bbox_inches='tight')
plt.close()

print(f"All analysis complete. Results in: {run_folder}")