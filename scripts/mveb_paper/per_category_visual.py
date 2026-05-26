import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
from pathlib import Path

def extract_family(model_name):
    """Heuristic to extract model family from HuggingFace-style names (org/model)."""
    model_str = str(model_name)
    if "/" in model_str:
        return model_str.split("/")[0]
    elif "-" in model_str:
        return model_str.split("-")[0]
    return "Other"

def plot_grouped_bar(plot_df, output_path="grouped_bar_spread.pdf", relative=False):
    """Creates a grouped bar chart showing the spread of top families across categories."""
    plt.figure(figsize=(14, 7))
    
    # Set a clean theme
    sns.set_theme(style="whitegrid")
    
    plot_df = plot_df.copy()
    ylabel = 'Max Score'
    
    if relative:
        max_per_cat = plot_df.groupby('Category')['Score'].transform('max')
        plot_df['Score'] = (plot_df['Score'] / max_per_cat) * 100
        ylabel = 'Relative Score (% of Category Max)'

    ax = sns.barplot(
        data=plot_df, 
        x='Category', 
        y='Score', 
        hue='Family',
        palette='viridis'
    )
    
    plt.xlabel('MTEB Category', fontsize=18, fontweight='bold')
    plt.ylabel(ylabel, fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(title='Model Family', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=15)
    
    plt.tight_layout()
    out_p = Path(output_path)
    plt.savefig(out_p.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(out_p.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"Grouped bar chart saved to {out_p.with_suffix('.pdf')} and .png")
    plt.show()

def plot_radar_chart(plot_df, categories, output_path="radar_chart_spread.pdf"):
    """Creates a radar/fan chart showing category spreads for the top families."""
    # Pivot data so rows are Families and columns are Categories
    pivot_df = plot_df.pivot(index='Family', columns='Category', values='Score').fillna(0)
    
    # Ensure columns match the exact order of the provided categories list
    pivot_df = pivot_df[categories]
    families = pivot_df.index.tolist()
    
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories, size=16, fontweight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=14)
    plt.ylim(0, 100) # Assuming scores are 0-100. Change to 0-1 if scores are 0.0-1.0
    
    # Plot each family
    colors = plt.cm.tab10.colors
    for i, family in enumerate(families):
        values = pivot_df.loc[family].tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=family, color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Model Family", fontsize=14, title_fontsize=15)
    
    plt.tight_layout()
    out_p = Path(output_path)
    plt.savefig(out_p.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(out_p.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to {out_p.with_suffix('.pdf')} and .png")
    plt.show()

def plot_size_vs_rank(plot_df, output_path="size_vs_rank.pdf"):
    """Creates a scatter plot showing Model Size vs Borda Rank."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create a copy and ensure target columns are numeric
    plot_df = plot_df.copy()
    plot_df['Model size'] = pd.to_numeric(plot_df['Model size'], errors='coerce')
    plot_df['Borda Rank'] = pd.to_numeric(plot_df['Borda Rank'], errors='coerce')
    plot_df = plot_df.dropna(subset=['Model size', 'Borda Rank'])
    
    ax = sns.scatterplot(
        data=plot_df, 
        x='Model size', 
        y='Borda Rank', 
        hue='Family',
        s=120,
        palette='tab10'
    )
    
    # Apply log scale with base 2 and add "B" suffix to the ticks
    plt.xscale('log', base=2)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}B'))
    plt.xticks(fontsize=14)
    
    # Invert Y-axis so rank 1 is at the top
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=14)
    
    plt.xlabel('Model Size', fontsize=18, fontweight='bold')
    plt.ylabel('Borda Rank', fontsize=18, fontweight='bold')
    plt.legend(title='Model Family', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=15)
    
    plt.tight_layout()
    out_p = Path(output_path)
    plt.savefig(out_p.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(out_p.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {out_p.with_suffix('.pdf')} and .png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate per-category visualizations.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the result CSV file")
    parser.add_argument("--plot_dir", type=str, default=".", help="Directory to save the PDF plots")
    parser.add_argument("--relative", action="store_true", help="Plot relative scores (percentage of max per category)")
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the data
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv_path}. Please check the path.")
        return

    # Identify the model and family columns
    model_col = 'Model Name' if 'Model Name' in df.columns else df.columns[0]
    
    if 'Model Family' in df.columns:
        df['Family'] = df['Model Family']
    else:
        df['Family'] = df[model_col].apply(extract_family)

    # 2. Define the target categories
    target_categories = [
        'Retr', 'QA', 'CLs', 'Clust', 'Pair', 'ZS'
    ]
    
    # Filter to only the categories actually present in the CSV
    available_categories = [cat for cat in target_categories if cat in df.columns]
    
    if not available_categories:
        print("Error: None of the target categories were found in the CSV columns.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Generate Model Size vs Rank scatter plot
    if 'Model size' in df.columns and 'Borda Rank' in df.columns:
        plot_size_vs_rank(df, output_path=plot_dir / "size_vs_rank.pdf")

    # 3. Melt dataframe for easier grouping
    melted_df = df.melt(
        id_vars=[model_col, 'Family'], 
        value_vars=available_categories,
        var_name='Category', 
        value_name='Score'
    )
    
    # Ensure scores are numeric
    melted_df['Score'] = pd.to_numeric(melted_df['Score'], errors='coerce')
    
    # 4. Find the best model per family, per category
    # Drop NaNs before finding the max to avoid errors
    clean_df = melted_df.dropna(subset=['Score'])
    idx_best = clean_df.groupby(['Category', 'Family'])['Score'].idxmax()
    best_per_family = clean_df.loc[idx_best]

    # Optional: Filter to top 5-6 families overall to keep the chart legible
    overall_family_scores = best_per_family.groupby('Family')['Score'].mean()
    top_families = overall_family_scores.nlargest(10).index
    plot_df = best_per_family[best_per_family['Family'].isin(top_families)]

    # If scores are on a 0.0 - 1.0 scale, multiply by 100 for better visualization
    if plot_df['Score'].max() <= 1.0:
        plot_df.loc[:, 'Score'] = plot_df['Score'] * 100

    # 5. Generate Visualizations
    plot_grouped_bar(plot_df, output_path=plot_dir / "grouped_bar_spread.pdf", relative=args.relative)
    plot_radar_chart(plot_df, available_categories, output_path=plot_dir / "radar_chart_spread.pdf")

if __name__ == "__main__":
    main()
