import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict

# === Define input and output directories ===
results_root = "/Users/nikalu/Downloads/53BP1_Results/Analysis_20250504_160804_53BP1v8"
analysis_output_root = "/Users/nikalu/Downloads/53BP1_Stats_Results"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
script_name = os.path.splitext(os.path.basename(__file__))[0]
run_folder = os.path.join(analysis_output_root, f'Analysis_{timestamp}_{script_name}')
os.makedirs(run_folder, exist_ok=True)

# === Store combined data per passage ===
combined_data = defaultdict(list)

# === Walk through all Results.csv files ===
for root, dirs, files in os.walk(results_root):
    for file in files:
        if file.lower().endswith(".csv"):
            csv_path = os.path.join(root, file)
            try:
                rel_path = os.path.relpath(csv_path, results_root)
                parts = rel_path.split(os.sep)
                if len(parts) < 2:
                    continue

                # Extract passage from parent folder name
                parent_folder = parts[0]
                passage_parts = parent_folder.split("_")
                passage = next((p for p in passage_parts if p.startswith("P")), None)
                if passage is None:
                    print(f"Could not extract passage from: {parent_folder}")
                    continue

                # Extract line from subfolder (e.g., "338 P19" → "338")
                line_folder = parts[1]
                line = line_folder.replace(passage, "").strip()

                # Create output folder
                output_folder = os.path.join(run_folder, passage, line)
                os.makedirs(output_folder, exist_ok=True)

                # Load and clean CSV
                df = pd.read_csv(csv_path)
                if '53BP1 foci' not in df.columns or 'Cell Number' not in df.columns:
                    print(f"Skipping {csv_path}: missing required columns.")
                    continue

                df['Foci per Cell'] = df['53BP1 foci'] / df['Cell Number']
                df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
                df.dropna(subset=['Foci per Cell'], inplace=True)

                # Add metadata
                df['Line'] = line
                df['Passage'] = passage
                combined_data[passage].append(df)

                # Save cleaned CSV
                df.to_csv(os.path.join(output_folder, f"{line}_Results.csv"), index=False)

            except Exception as e:
                print(f"Failed: {csv_path} → {e}")

# === Per passage plots and summary ===
summary_rows = []
all_combined_df = []

for passage, dfs in combined_data.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    all_combined_df.append(combined_df)
    passage_folder = os.path.join(run_folder, passage)
    os.makedirs(passage_folder, exist_ok=True)

    lines = sorted(combined_df['Line'].unique())
    data_per_line = [combined_df[combined_df['Line'] == line]['Foci per Cell'] for line in lines]

    # Boxplot with means and medians
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data_per_line, tick_labels=lines, patch_artist=True, showmeans=True)
    for median in box['medians']:
        median.set(color='red', linewidth=2)
    for mean in box['means']:
        mean.set(marker='o', markerfacecolor='blue', markeredgecolor='black', markersize=6)
    plt.title(f"{passage} - Foci per Cell (Boxplot per Line)")
    plt.xlabel("Line")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()
    plt.savefig(os.path.join(passage_folder, f"{passage}_Boxplot_Per_Line.png"))
    plt.close()

    # Violin plot
    plt.figure(figsize=(8, 6))
    plt.violinplot(data_per_line, showmeans=True, showmedians=True, showextrema=True)
    plt.xticks(ticks=np.arange(1, len(lines) + 1), labels=lines)
    plt.title(f"{passage} - Foci per Cell (Violin Plot)")
    plt.xlabel("Line")
    plt.ylabel("Foci per Cell")
    plt.tight_layout()
    plt.savefig(os.path.join(passage_folder, f"{passage}_ViolinPlot_Per_Line.png"))
    plt.close()

    # Histogram overlay
    plt.figure(figsize=(8, 6))
    for line in lines:
        subset = combined_df[combined_df['Line'] == line]['Foci per Cell']
        plt.hist(subset, bins=20, alpha=0.5, label=line, edgecolor='black')
    plt.title(f"{passage} - Foci per Cell (Histogram Overlay)")
    plt.xlabel("Foci per Cell")
    plt.ylabel("Frequency")
    plt.legend(title="Line")
    plt.tight_layout()
    plt.savefig(os.path.join(passage_folder, f"{passage}_HistogramOverlay_Per_Line.png"))
    plt.close()

    # Summary statistics
    for line in lines:
        subset = combined_df[combined_df['Line'] == line]['Foci per Cell']
        summary_rows.append({
            'Passage': passage,
            'Line': line,
            'N': len(subset),
            'Mean': subset.mean(),
            'Median': subset.median(),
            'Std': subset.std(),
            'Min': subset.min(),
            'Max': subset.max(),
            'Q1': subset.quantile(0.25),
            'Q3': subset.quantile(0.75),
            'IQR': subset.quantile(0.75) - subset.quantile(0.25),
        })

    print(f"Plots saved for {passage}")

# === Save overall summary CSV ===
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(run_folder, "Summary_Statistics_Per_Passage.csv"), index=False)

# === Global combined comparison across all passages and lines ===
all_df = pd.concat(all_combined_df, ignore_index=True)
all_df['Group'] = all_df['Passage'] + " - " + all_df['Line']
groups = sorted(all_df['Group'].unique())
data_per_group = [all_df[all_df['Group'] == grp]['Foci per Cell'] for grp in groups]

# Global boxplot
plt.figure(figsize=(max(10, len(groups)), 6))
box = plt.boxplot(data_per_group, tick_labels=groups, patch_artist=True, showmeans=True)
for median in box['medians']:
    median.set(color='red', linewidth=2)
for mean in box['means']:
    mean.set(marker='o', markerfacecolor='blue', markeredgecolor='black', markersize=5)
plt.xticks(rotation=90)
plt.title("All Passages & Lines - Foci per Cell (Boxplot)")
plt.ylabel("Foci per Cell")
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Combined_Boxplot_All_Passages_Lines.png"))
plt.close()

# Global violin plot
plt.figure(figsize=(max(10, len(groups)), 6))
plt.violinplot(data_per_group, showmeans=True, showmedians=True, showextrema=True)
plt.xticks(ticks=np.arange(1, len(groups) + 1), labels=groups, rotation=90)
plt.title("All Passages & Lines - Foci per Cell (Violin Plot)")
plt.ylabel("Foci per Cell")
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Combined_ViolinPlot_All_Passages_Lines.png"))
plt.close()

# Global histogram overlay
plt.figure(figsize=(10, 6))
for group in groups:
    subset = all_df[all_df['Group'] == group]['Foci per Cell']
    plt.hist(subset, bins=20, alpha=0.4, label=group, edgecolor='black')
plt.title("All Passages & Lines - Foci per Cell (Histogram Overlay)")
plt.xlabel("Foci per Cell")
plt.ylabel("Frequency")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(run_folder, "Combined_HistogramOverlay_All_Passages_Lines.png"))
plt.close()

print(f"\nAnalysis complete! All plots and statistics saved in: {run_folder}")