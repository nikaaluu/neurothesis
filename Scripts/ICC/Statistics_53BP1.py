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
from scipy.stats import f_oneway, levene, kruskal, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from pingouin import welch_anova, pairwise_gameshowell
from scikit_posthocs import posthoc_dunn
import seaborn as sns
import matplotlib.gridspec as gridspec
import scikit_posthocs as sp

# === Step 1: Define input and output directories ===
input_csv = "/Users/nikalu/Downloads/summary_per_well.csv"
analysis_output_root = "/Users/nikalu/Downloads/53BP1_Stats_Results"
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
    plt.title(f"{passage} - Foci per Cell (Boxplot)")
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
    plt.title(f"{passage} - Foci per Cell (Violin)")
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
    plt.title(f"{passage} Foci per Cell (Histogram)")
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

# Perform Tukey's post-hoc test for the two-way ANOVA
tukey = pairwise_tukeyhsd(
    endog=all_df['Foci_per_Cell'],
    groups=all_df['Passage'] + " - " + all_df['Line'],
    alpha=0.05
)

# Save Tukey results to a CSV file
tukey_results = []
for row in tukey.summary().data[1:]:
    p_value = max(0, min(row[5], 1))  # Ensure p-value is between 0 and 1
    tukey_results.append({
        'Comparison': f"{row[0]} vs {row[1]}",
        'p-value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

pd.DataFrame(tukey_results).to_csv(os.path.join(run_folder, "Tukey_Two_Way_ANOVA.csv"), index=False)

# Create a combined grouping variable for Passage and Line
all_df['Passage_Line'] = all_df['Passage'] + " - " + all_df['Line']

# Perform Games-Howell post-hoc test for the two-way ANOVA
gameshowell_results_anova = pairwise_gameshowell(
    data=all_df,
    dv='Foci_per_Cell',  # Dependent variable
    between='Passage_Line'  # Combined grouping variable
)

# Save Games-Howell results to a CSV file
gameshowell_results_anova.to_csv(
    os.path.join(run_folder, "Games_Howell_Two_Way_ANOVA.csv"), index=False
)

print("Games-Howell post-hoc test for two-way ANOVA complete. Results saved.")

# Reorganize data to start from specific passages
desired_passages = ['P5', 'P6', 'P7', 'P19', 'P20', 'P21']
all_df = all_df[all_df['Passage'].isin(desired_passages)]
all_df['Passage'] = pd.Categorical(all_df['Passage'], categories=desired_passages, ordered=True)
all_df = all_df.sort_values(by=['Passage', 'Line'])

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

print(all_df.groupby(['Passage', 'Line'])['Foci_per_Cell'].var())

# Collect Levene's and Shapiro-Wilk test results
normality_homogeneity_results = []

for passage in passage_order:
    grp = all_df[all_df['Passage'] == passage]
    
    # Shapiro-Wilk test for normality
    shapiro_pval = shapiro(grp['Foci_per_Cell'])[1]
    
    # Levene's test for homogeneity of variances
    levene_pval = levene(*[grp[grp['Line'] == ln]['Foci_per_Cell'] for ln in line_order if ln in grp['Line'].unique()])[1]
    
    # Append results
    normality_homogeneity_results.append({
        'Passage': passage,
        'Shapiro-p-value': shapiro_pval,
        'Levene-p-value': levene_pval
    })

# Save the results to a CSV file
pd.DataFrame(normality_homogeneity_results).to_csv(
    os.path.join(run_folder, "Normality_Homogeneity_Tests.csv"), index=False
)

# Perform Shapiro-Wilk and Levene's tests for cell lines
normality_homogeneity_results_lines = []

for line in line_order:
    grp = all_df[all_df['Line'] == line]
    
    # Shapiro-Wilk test for normality
    shapiro_pval = shapiro(grp['Foci_per_Cell'])[1] if len(grp) > 3 else np.nan  # Shapiro requires at least 3 samples
    
    # Levene's test for homogeneity of variances across passages
    data = [grp[grp['Passage'] == p]['Foci_per_Cell'] for p in passage_order if p in grp['Passage'].unique()]
    levene_pval = levene(*data)[1] if len(data) > 1 else np.nan  # Levene requires at least 2 groups
    
    # Append results
    normality_homogeneity_results_lines.append({
        'Line': line,
        'Shapiro-p-value': shapiro_pval,
        'Levene-p-value': levene_pval
    })

# Save the results to a CSV file
pd.DataFrame(normality_homogeneity_results_lines).to_csv(
    os.path.join(run_folder, "Normality_Homogeneity_Tests_Cell_Lines.csv"), index=False
)

print("Shapiro-Wilk and Levene's tests for cell lines complete. Results saved.")

# Perform Kruskal-Wallis and Games-Howell tests for all passages and cell lines
kruskal_results = []
gameshowell_results = []

for passage in passage_order:
    grp = all_df[all_df['Passage'] == passage]
    
    # Prepare data for Kruskal-Wallis test
    data = [grp[grp['Line'] == ln]['Foci_per_Cell'] for ln in line_order if ln in grp['Line'].unique()]
    
    # Kruskal-Wallis test
    if len(data) > 1:  # Ensure there are at least two groups to compare
        stat, pval = kruskal(*data)
        kruskal_results.append({'Passage': passage, 'Kruskal-Wallis_stat': stat, 'p-value': pval})
    
    # Games-Howell post-hoc test
    if len(grp['Line'].unique()) > 1:  # Ensure there are at least two groups to compare
        posthoc = pairwise_gameshowell(data=grp, dv='Foci_per_Cell', between='Line')
        posthoc['Passage'] = passage  # Add passage information to the results
        gameshowell_results.append(posthoc)

# Save Kruskal-Wallis results to a CSV file
pd.DataFrame(kruskal_results).to_csv(
    os.path.join(run_folder, "Kruskal_Wallis_Results.csv"), index=False
)

# Save Games-Howell results to a CSV file
if gameshowell_results:
    pd.concat(gameshowell_results).to_csv(
        os.path.join(run_folder, "Games_Howell_Results.csv"), index=False
    )

print("Kruskal-Wallis and Games-Howell tests complete. Results saved.")

# Perform Kruskal-Wallis test for differences amongst cell lines
kruskal_line_results = []

for line in line_order:
    grp = all_df[all_df['Line'] == line]
    
    # Prepare data for Kruskal-Wallis test
    data = [grp[grp['Passage'] == p]['Foci_per_Cell'] for p in passage_order if p in grp['Passage'].unique()]
    
    # Kruskal-Wallis test
    if len(data) > 1:  # Ensure there are at least two groups to compare
        stat, pval = kruskal(*data)
        kruskal_line_results.append({'Line': line, 'Kruskal-Wallis_stat': stat, 'p-value': pval})

# Save Kruskal-Wallis results for cell lines to a CSV file
pd.DataFrame(kruskal_line_results).to_csv(
    os.path.join(run_folder, "Kruskal_Wallis_Results_Cell_Lines.csv"), index=False
)

print("Kruskal-Wallis test for cell lines complete. Results saved.")

# Perform Games-Howell and Dunn's post-hoc tests for Kruskal-Wallis results
gameshowell_posthoc_results = []
dunn_posthoc_results = []

# Post-hoc for passages within each line
for line in line_order:
    grp = all_df[all_df['Line'] == line]
    if len(grp['Passage'].unique()) > 1:  # Ensure there are at least two groups
        # Games-Howell post-hoc test
        gameshowell = pairwise_gameshowell(data=grp, dv='Foci_per_Cell', between='Passage')
        gameshowell['Line'] = line
        gameshowell_posthoc_results.append(gameshowell)

        # Dunn's post-hoc test
        dunn = sp.posthoc_dunn(grp, val_col='Foci_per_Cell', group_col='Passage', p_adjust='bonferroni')
        dunn['Line'] = line
        dunn_posthoc_results.append(dunn)

# Post-hoc for lines within each passage
for passage in passage_order:
    grp = all_df[all_df['Passage'] == passage]
    if len(grp['Line'].unique()) > 1:  # Ensure there are at least two groups
        # Games-Howell post-hoc test
        gameshowell = pairwise_gameshowell(data=grp, dv='Foci_per_Cell', between='Line')
        gameshowell['Passage'] = passage
        gameshowell_posthoc_results.append(gameshowell)

        # Dunn's post-hoc test
        dunn = sp.posthoc_dunn(grp, val_col='Foci_per_Cell', group_col='Line', p_adjust='bonferroni')
        dunn['Passage'] = passage
        dunn_posthoc_results.append(dunn)

# Save Games-Howell results to a CSV file
if gameshowell_posthoc_results:
    pd.concat(gameshowell_posthoc_results).to_csv(
        os.path.join(run_folder, "Games_Howell_Posthoc_Kruskal_Wallis.csv"), index=False
    )

# Save Dunn's test results to a CSV file
if dunn_posthoc_results:
    pd.concat(dunn_posthoc_results).to_csv(
        os.path.join(run_folder, "Dunn_Posthoc_Kruskal_Wallis.csv"), index=False
    )

print("Games-Howell and Dunn's post-hoc tests for Kruskal-Wallis complete. Results saved.")

# Perform Dunn's test for all passages across all lines
dunn_passages = sp.posthoc_dunn(
    all_df, val_col='Foci_per_Cell', group_col='Passage', p_adjust='bonferroni'
)

# Save the results to a CSV file
dunn_passages.to_csv(os.path.join(run_folder, "Dunn_Posthoc_All_Passages.csv"))

print("Dunn's test for all passages complete. Results saved.")

# Perform Dunn's test for all lines across all passages
dunn_lines = sp.posthoc_dunn(
    all_df, val_col='Foci_per_Cell', group_col='Line', p_adjust='bonferroni'
)

# Save the results to a CSV file
dunn_lines.to_csv(os.path.join(run_folder, "Dunn_Posthoc_All_Lines.csv"))

print("Dunn's test for all lines complete. Results saved.")

# Perform Dunn's test for all passages across all lines with Holm correction
dunn_passages_holm = sp.posthoc_dunn(
    all_df, val_col='Foci_per_Cell', group_col='Passage', p_adjust='holm'
)

# Save the results to a CSV file
dunn_passages_holm.to_csv(
    os.path.join(run_folder, "Dunn_Posthoc_All_Passages_Holm.csv")
)

print("Dunn's test for all passages with Holm correction complete. Results saved.")

# Perform Dunn's test for all lines across all passages with Holm correction
dunn_lines_holm = sp.posthoc_dunn(
    all_df, val_col='Foci_per_Cell', group_col='Line', p_adjust='holm'
)

# Save the results to a CSV file
dunn_lines_holm.to_csv(
    os.path.join(run_folder, "Dunn_Posthoc_All_Lines_Holm.csv")
)

print("Dunn's test for all lines with Holm correction complete. Results saved.")

# === Step 7: QQ Plots and Residual Plots ===

# Generate QQ plots and residual plots for the two-way ANOVA model
residuals = model.resid  # Residuals from the two-way ANOVA model
fitted_values = model.fittedvalues  # Fitted values from the model

# QQ Plot
plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
plt.title("QQ Plot of Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "QQ_Plot_Residuals.png"))
plt.close()

# Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.7, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Residuals_vs_Fitted.png"))
plt.close()

print("QQ plot and residual plot complete. Results saved.")

print(f"All analysis complete. Results in: {run_folder}")