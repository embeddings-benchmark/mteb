#!/usr/bin/env python3
"""
Print the domain distribution for each of the 4 MAEB benchmarks.
"""

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mteb import get_benchmark


MAEB_BENCHMARKS = [
    "MAEB",
    "MAEB+",
    "MAEB(audio-only)",
    "MAEB(extended)",
]


def get_domain_counts(benchmark_name: str) -> tuple[Counter, int]:
    """Extract and count domains from all tasks in a benchmark."""
    benchmark = get_benchmark(benchmark_name)

    domain_counts = Counter()

    for task in benchmark.tasks:
        domains = task.metadata.domains

        if domains:
            for domain in domains:
                domain_counts[domain] += 1
        else:
            domain_counts["None"] += 1

    return domain_counts, len(benchmark.tasks)


def print_domain_distribution(benchmark_name: str) -> None:
    """Print the domain distribution for a benchmark."""
    domain_counts, num_tasks = get_domain_counts(benchmark_name)

    print(f"\n{'=' * 60}")
    print(f"{benchmark_name} ({num_tasks} tasks)")
    print("=" * 60)

    total_domain_pairs = sum(domain_counts.values())
    print(f"Total task-domain pairs: {total_domain_pairs}")
    print(f"Unique domains: {len(domain_counts)}")
    print()

    print("Domain distribution (sorted by count):")
    print("-" * 40)
    for domain, count in domain_counts.most_common():
        percentage = (count / total_domain_pairs) * 100
        print(f"  {domain:20s} {count:3d} ({percentage:5.1f}%)")


def plot_domain_distribution() -> None:
    """Create a figure with 3 side-by-side pie charts for MAEB, MAEB+, and MAEB(audio-only)."""
    benchmarks_to_plot = ["MAEB+", "MAEB", "MAEB(audio-only)"]

    # Collect domain counts for each benchmark
    all_domain_counts = {}
    for benchmark_name in benchmarks_to_plot:
        domain_counts, num_tasks = get_domain_counts(benchmark_name)
        all_domain_counts[benchmark_name] = (domain_counts, num_tasks)

    # Sort domains by total count across all benchmarks for consistent ordering
    domain_totals = Counter()
    for domain_counts, _ in all_domain_counts.values():
        domain_totals.update(domain_counts)
    sorted_domains = [d for d, _ in domain_totals.most_common()]

    # Create a color map for consistent colors across pie charts
    cmap = plt.cm.tab20
    domain_colors = {
        domain: cmap(i / len(sorted_domains)) for i, domain in enumerate(sorted_domains)
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, benchmark_name in zip(axes, benchmarks_to_plot):
        domain_counts, num_tasks = all_domain_counts[benchmark_name]

        # Prepare data in consistent order (include all domains for consistent colors)
        sizes = []
        colors = []
        for domain in sorted_domains:
            count = domain_counts.get(domain, 0)
            sizes.append(count)
            colors.append(domain_colors[domain])

        # Create pie chart without labels (we'll use a shared legend)
        # Filter out zero-sized wedges for cleaner appearance
        ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        ax.set_title(f"{benchmark_name}\n({num_tasks} tasks)")

    # Create legend patches for all domains
    legend_patches = [
        Patch(facecolor=domain_colors[d], label=d) for d in sorted_domains
    ]

    # Create a single legend for all charts
    fig.legend(
        handles=legend_patches,
        title="Domains",
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
    )

    # plt.suptitle("MAEB Domain Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("maeb_domain_distribution.png", dpi=150, bbox_inches="tight")
    plt.savefig("maeb_domain_distribution.pdf", bbox_inches="tight")
    print(
        "\nFigure saved to maeb_domain_distribution.png and maeb_domain_distribution.pdf"
    )


def main():
    """Print domain distribution for all MAEB benchmarks."""
    print("MAEB Domain Distribution Analysis")
    print("=" * 60)

    for benchmark_name in MAEB_BENCHMARKS:
        try:
            print_domain_distribution(benchmark_name)
        except Exception as e:
            print(f"\nError processing {benchmark_name}: {e}")

    print("\n" + "=" * 60)
    print("Analysis complete.")

    # Generate the pie chart figure
    plot_domain_distribution()


if __name__ == "__main__":
    main()
