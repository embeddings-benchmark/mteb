#!/usr/bin/env python3
"""
Plot the language counts of MAEB+ collection in a bar graph and save as PDF.
"""

import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from mteb import get_benchmark


def get_language_counts():
    """Extract and count languages from all MAEB+ tasks."""
    # Load the main MAEB benchmark
    print("Loading MAEB+ benchmark...")
    maeb_benchmark = get_benchmark("MAEB+")

    # Extract all languages from all tasks
    all_languages = []

    for task in maeb_benchmark.tasks:
        # Get languages from task metadata
        task_languages = task.metadata.languages

        if task_languages:
            # Process each language
            for lang in task_languages:
                # Take the primary language code (before any dash)
                # e.g., "en-US" -> "en"
                primary_lang = lang.split("-")[0] if lang else None
                if primary_lang:
                    all_languages.append(primary_lang)

    # Count occurrences of each language
    language_counts = Counter(all_languages)

    return language_counts


def plot_language_distribution(
    language_counts, output_file="maeb_language_distribution.pdf"
):
    """Create a bar plot of language distribution and save as PDF."""

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(language_counts, orient="index", columns=["Count"])
    df = df.sort_values("Count", ascending=False)

    print(f"\nLanguage distribution ({len(df)} unique languages):")
    print(df.head(20))  # Show top 20 languages

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 5))

    # Create bar plot with slightly darker blue color
    bars = ax.bar(range(len(df)), df["Count"].values, color="skyblue")

    # Customize the plot
    ax.set_xlabel("Language Code", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Tasks", fontsize=12, fontweight="bold")
    ax.set_title(
        "Language Distribution in MAEB+ Collection", fontsize=14, fontweight="bold"
    )

    # Set x-axis labels with 90 degree rotation
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=90, ha="center", va="top")

    # Remove gaps on the sides by setting x-axis limits
    ax.set_xlim(-0.5, len(df) - 0.5)

    # Add grid for better readability
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add statistics text box
    stats_text = f"Total Languages: {len(df)}\n"
    stats_text += f"Total Task-Language Pairs: {df['Count'].sum()}\n"
    stats_text += f"Average Tasks per Language: {df['Count'].mean():.1f}\n"
    stats_text += f"Most Common: {df.index[0]} ({df['Count'].iloc[0]} tasks)"

    # Place text box in upper right
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save as PDF
    plt.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as: {output_file}")

    # Also save as PNG for quick viewing
    png_file = output_file.replace(".pdf", ".png")
    plt.savefig(png_file, format="png", dpi=150, bbox_inches="tight")
    print(f"Also saved as: {png_file}")

    plt.show()

    return df


def main():
    """Main function to generate language distribution plot."""
    try:
        # Get language counts
        language_counts = get_language_counts()

        if not language_counts:
            print("No language data found in MAEB+ tasks.")
            return

        print(f"Found {len(language_counts)} unique languages across MAEB+ tasks")

        # Create and save the plot
        df_stats = plot_language_distribution(language_counts)

        # Print detailed statistics
        print("\n" + "=" * 50)
        print("DETAILED STATISTICS")
        print("=" * 50)
        print(f"Total unique languages: {len(language_counts)}")
        print(f"Total task-language pairs: {sum(language_counts.values())}")
        print(
            f"Average tasks per language: {sum(language_counts.values()) / len(language_counts):.2f}"
        )
        print(f"\nTop 10 languages by task count:")
        for i, (lang, count) in enumerate(language_counts.most_common(10), 1):
            print(f"  {i:2d}. {lang:5s} - {count:3d} tasks")

        # Languages with only 1 task
        single_task_langs = [
            lang for lang, count in language_counts.items() if count == 1
        ]
        if single_task_langs:
            print(f"\nLanguages with only 1 task ({len(single_task_langs)} total):")
            print(f"  {', '.join(sorted(single_task_langs))}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
