# ======================================================
# Author: Nika Lu Yang
# Date: April 2025
# Bachelor Thesis Research (BTR): "Unraveling the Impact of Aging on Parkinson’s Disease-derived Neuroepithelial Stem Cells"
# Maastricht Science Programme (MSP)
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Step 1: Define input and output directories ===
input_csv  = "results.csv"            
output_csv = "summary_per_well.csv"
plot_dir   = "plots_per_passage"
os.makedirs(plot_dir, exist_ok=True)

# === Step 2: Load and clean data ===
df = pd.read_csv(input_csv)

# 1) Exclude bad images
df = df[df['bad'] == False]

# 2) Strip whitespace from identifiers
df['cell_line'] = df['cell_line'].str.strip()
df['well']      = df['well'].str.strip()
df['passage']   = df['passage'].str.strip()

# 3) Convert long passage names ("Passage_19_…") to "P19"
#    Extract the digits after "Passage_" and prefix with "P"
df['passage'] = (
    df['passage']
      .str.extract(r'Passage_(\d+)', expand=False)  # pull out "19"
      .apply(lambda num: f'P{num}')                 # make "P19"
)

# === Group and aggregate per well ===
grouped = (
    df
    .groupby(['passage', 'cell_line', 'well'], as_index=False)
    .agg({'cell_count': 'sum', 'foci_count': 'sum'})
)
grouped['foci_per_cell'] = grouped['foci_count'] / grouped['cell_count']

# === Step 3: Save the per-well summary CSV ===
grouped.to_csv(output_csv, index=False)

# === Step 4: Plot per-passage ===
# ensure passages are plotted in sorted order P5, P6, P7, P19,…
passages = sorted(grouped['passage'].unique(), key=lambda p: int(p.lstrip('P')))

for p in passages:
    sub = grouped[grouped['passage'] == p]

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=sub,
        x='cell_line',
        y='foci_per_cell'
    )
    sns.stripplot(
        data=sub,
        x='cell_line',
        y='foci_per_cell',
        color='black',
        alpha=0.5,
        jitter=True
    )

    plt.title(f"Foci per Cell - {p}")
    plt.xlabel("Cell Line")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"{p}_foci_per_cell.png"))
    plt.close()