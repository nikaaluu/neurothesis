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
input_csv  = "/Users/nikalu/Downloads/Results_53BP1.csv"            
output_csv = "/Users/nikalu/Downloads/summary_per_well_.csv"
plot_dir   = "/Users/nikalu/Downloads/plots_per_passage"
os.makedirs(plot_dir, exist_ok=True)

# === Step 2: Load and clean data ===
df = pd.read_csv(input_csv)

# 1) Exclude bad images
df = df[df['bad'] == False]

# 2) Strip whitespace from identifiers
df['cell_line'] = df['cell_line'].str.strip()
df['well']      = df['well'].str.strip()
df['passage']   = df['passage'].str.strip()

# Clean the cell_line column
df['cell_line'] = df['cell_line'].str.strip()  # Remove leading/trailing spaces
df['cell_line'] = df['cell_line'].str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space
df['cell_line'] = df['cell_line'].str.upper()  # Standardize to uppercase
df['cell_line'] = df['cell_line'].replace({
    'T4.13GC': 'T4.13 GC',
    'T4.13M': 'T4.13 M',
    'T4.13 GC ': 'T4.13 GC',  # Handle trailing spaces
    'T4.13 M ': 'T4.13 M'    # Handle trailing spaces
})

# Verify unique values
print("Unique cell lines after cleaning:", df['cell_line'].unique())

# 3) Convert long passage names ("Passage_19_…") to "P19"
#    Extract the digits after "Passage_" and prefix with "P"
df['passage'] = (
    df['passage']
      .str.extract(r'Passage_(\d+)', expand=False)  # pull out "19"
      .apply(lambda num: f'P{num}')                 # make "P19"
)

# Check unique values in the cell_line column
print(df['cell_line'].unique())

# === Step 3: Group and aggregate per well ===
grouped = (
    df
    .groupby(['passage', 'cell_line', 'well'], as_index=False)
    .agg({'cell_count': 'sum', 'foci_count': 'sum'})
)
grouped['foci_per_cell'] = grouped['foci_count'] / grouped['cell_count']

# Clean the grouped DataFrame
grouped['cell_line'] = grouped['cell_line'].str.strip()
grouped['cell_line'] = grouped['cell_line'].str.replace(r'\s+', ' ', regex=True)
grouped['cell_line'] = grouped['cell_line'].str.upper()
grouped['cell_line'] = grouped['cell_line'].replace({
    'T4.13GC': 'T4.13 GC',
    'T4.13M': 'T4.13 M',
    'T4.13 GC ': 'T4.13 GC',
    'T4.13 M ': 'T4.13 M'
})

# Verify unique values in grouped
print("Unique cell lines in grouped after cleaning:", grouped['cell_line'].unique())

# === Step 4: Save the per-well summary CSV ===
grouped.to_csv(output_csv, index=False)

# === Step 5: Plotting ===
# Thsis ensures passages are plotted in sorted order P5, P6, P7, P19,…
passages = sorted(grouped['passage'].unique(), key=lambda p: int(p.lstrip('P')))

for p in passages:
    sub = grouped[grouped['passage'] == p]

    # 1) Plotting foci per cell
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=sub,
        x='cell_line',
        y='foci_per_cell',
        hue='cell_line',  # Assign the x variable to hue
        palette='Set2',
        dodge=False,      # Prevents splitting the boxes by hue
        legend=False      # Suppresses the legend
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
    
    # 2) Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=sub,
        x='cell_line',
        y='foci_per_cell',
        hue='cell_line',  # Assign the x variable to hue
        inner='quartile',  # Show quartiles
        palette='Set2',
        legend=False       # Suppress the legend
    )
    plt.title(f"Violin Plot of Foci per Cell - {p}")
    plt.xlabel("Cell Line")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{p}_foci_violin.png"))
    plt.close()

    # 3) Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=sub,
        x='foci_per_cell',
        bins=20,
        kde=True,  # Add a density curve
        hue='cell_line',  # Differentiate by cell line
        palette='Set2'
    )
    plt.title(f"Histogram of Foci per Cell - {p}")
    plt.xlabel("Foci per Cell")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{p}_foci_histogram.png"))
    plt.close()

    # 4) Bar Plot with error bars
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=sub,
        x='cell_line',
        y='foci_per_cell',
        hue='cell_line',  # Assign the x variable to hue
        palette='Set2',
        errorbar='sd',    # Use errorbar='sd' instead of ci='sd'
        legend=False      # Suppress the legend
    )
    plt.title(f"Mean Foci per Cell with Error Bars - {p}")
    plt.xlabel("Cell Line")
    plt.ylabel("Mean Foci per Cell (±SD)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{p}_foci_bar.png"))
    plt.close()

    # 5) Swarm Plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(
        data=sub,
        x='cell_line',
        y='foci_per_cell',
        color='blue',
        alpha=0.7
    )
    plt.title(f"Swarm Plot of Foci per Cell - {p}")
    plt.xlabel("Cell Line")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{p}_foci_swarm.png"))
    plt.close()

    # 6) Facet Grid
    g = sns.FacetGrid(sub, col='cell_line', col_wrap=3, height=4, sharex=True, sharey=True)
    g.map(sns.histplot, 'foci_per_cell', bins=20, kde=True)
    g.set_titles("{col_name}")
    g.set_axis_labels("Foci per Cell", "Frequency")
    g.tight_layout()
    g.savefig(os.path.join(plot_dir, f"{p}_foci_facetgrid.png"))

for cell_line in grouped['cell_line'].unique():
    sub = grouped[grouped['cell_line'] == cell_line]

    # Violin Plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=sub,
        x='passage',
        y='foci_per_cell',
        hue='passage',       # Differentiate by passage
        palette='Set2',
        inner='quartile',    # Show quartiles inside the violins
        dodge=True,          # Split violins by hue
        density_norm='width' # Adjust the width of violins based on the number of observations
    )
    plt.title(f"Violin Plot of Foci per Cell for {cell_line}")
    plt.xlabel("Passage")
    plt.ylabel("Foci per Cell")
    
    # Adjust x-axis limits to reduce spacing
    plt.xlim(-0.5, len(sub['passage'].unique()) - 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{cell_line}_violin_passages.png"))
    plt.close()

    # Box Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=sub,
        x='passage',
        y='foci_per_cell',
        hue='passage',       # Differentiate by passage
        palette='Set2',
        dodge=True,           # Split boxes by hue
        width=0.8            # Reduce the width of the violins

    )
    plt.title(f"Box Plot of Foci per Cell for {cell_line}")
    plt.xlabel("Passage")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{cell_line}_box_passages.png"))
    plt.close()

     # Histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(
        data=sub,
        x='foci_per_cell',
        hue='passage',       # Differentiate by passage
        bins=20,             # Number of bins
        kde=True,            # Add a density curve
        palette='Set2',      # Use a color palette
        alpha=0.7            # Transparency for overlapping bars
    )
    plt.title(f"Histogram of Foci per Cell for {cell_line}")
    plt.xlabel("Foci per Cell")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{cell_line}_histogram_passages.png"))
    plt.close()

    # Combine A13 and 338
    sub_a13_338 = grouped[grouped['cell_line'].isin(['A13', '338'])]

    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=sub_a13_338,
        x='passage',
        y='foci_per_cell',
        hue='cell_line',       # Differentiate between A13 and 338
        palette='Set2',
        inner='quartile',      # Show quartiles inside the violins
        dodge=True,            # Split violins by hue
        density_norm='width'   # Adjust the width of violins based on the number of observations
    )
    plt.title("Violin Plot of Foci per Cell for A13 and 338")
    plt.xlabel("Passage")
    plt.ylabel("Foci per Cell")
    plt.legend(title="Cell Line", loc="upper right", bbox_to_anchor=(1.15, 1))  # Adjust legend position
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "A13_338_violin_passages.png"))
    plt.close()

    # Combine T4.13 GC and T4.13 M
    sub_t4 = grouped[grouped['cell_line'].isin(['T4.13 GC', 'T4.13 M'])]

    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=sub_t4,
        x='passage',
        y='foci_per_cell',
        hue='cell_line',       # Differentiate between T4.13 GC and T4.13 M
        palette='Set2',
        inner='quartile',      # Show quartiles inside the violins
        dodge=True,            # Split violins by hue
        density_norm='width'   # Adjust the width of violins based on the number of observations
    )
    plt.title("Violin Plot of Foci per Cell for T4.13 GC and T4.13 M")
    plt.xlabel("Passage")
    plt.ylabel("Foci per Cell")
    plt.legend(title="Cell Line", loc="upper right", bbox_to_anchor=(1.15, 1))  # Adjust legend position
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "T4.13GC_T4.13M_violin_passages.png"))
    plt.close()

    # Combine all four cell lines
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=grouped,
        x='passage',
        y='foci_per_cell',
        hue='cell_line',       # Differentiate between all four cell lines
        palette='Set2',
        inner='quartile',      # Show quartiles inside the violins
        dodge=True,            # Split violins by hue
        density_norm='width'   # Adjust the width of violins based on the number of observations
    )
    plt.title("Violin Plot of Foci per Cell for All Cell Lines")
    plt.xlabel("Passage")
    plt.ylabel("Foci per Cell")
    plt.xticks(ticks=range(len(grouped['passage'].unique())), labels=grouped['passage'].unique(), fontsize=12)
    plt.legend(title="Cell Line", loc="upper right")  # Adjust the location to "upper left"
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "All_Cell_Lines_violin_passages.png"))
    plt.close()

    passage_order = ['P5', 'P6', 'P7', 'P19', 'P20', 'P21']

    # Combine all four cell lines with box and whiskers inside the violins
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=grouped,
        x='passage',
        y='foci_per_cell',
        hue='cell_line',       # Differentiate between all four cell lines
        palette='Set2',
        inner='box',           # Add box and whiskers inside the violins
        dodge=True,            # Split violins by hue
        density_norm='width',   # Adjust the width of violins based on the number of observations
        order=passage_order  # Ensure the order of passages
    )
    plt.title("Violin Plot of Foci per Cell for All Cell Lines (with Box and Whiskers)")
    plt.xlabel("Passage")
    plt.ylabel("Foci per Cell")

    # Add a legend inside the plot
    plt.legend(title="Cell Line", loc="upper left")  # Place the legend inside the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "All_Cell_Lines_violin_box_whiskers.png"))
    plt.close()

# Define the desired order of cell lines
cell_line_order = ['A13', '338', 'T4.13 GC', 'T4.13 M']

# Create a pivot table for the heatmap
heatmap_data = grouped.pivot_table(
    index='passage', 
    columns='cell_line', 
    values='foci_per_cell', 
    aggfunc='mean'
).reindex(columns=cell_line_order)

# Fill missing values with 0
heatmap_data = heatmap_data.fillna(0)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data, 
    annot=True,          # Display values inside the boxes
    fmt=".2f",           # Format the numbers to 2 decimal places
    cmap="coolwarm",     # Use a diverging colormap
    cbar_kws={'label': 'Mean Foci per Cell'}  # Label for the color bar
)
plt.title("Mean Foci per Cell")
plt.xlabel("Cell Line")
plt.ylabel("Passage")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "heatmap_foci_per_cell.png"))
plt.close()

