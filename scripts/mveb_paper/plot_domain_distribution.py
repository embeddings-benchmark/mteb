"""Bar plot of domain coverage across the 184 MVEB+ tasks."""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

warnings.filterwarnings("ignore")

import mteb  # noqa: E402

OUT_DIR = Path(__file__).parent


def main(fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(mteb.get_tasks(modalities=["video"], exclude_beta=False))
    print(f"Loaded {len(tasks)} MVEB+ tasks")

    # Task-level domain counts (a task with multiple domain tags counts in each)
    task_counter = Counter()
    for t in tasks:
        domains = getattr(t.metadata, "domains", None) or []
        for d in domains:
            task_counter[d] += 1

    # Dataset-level (deduplicated by dataset path) counts
    dataset_domains: dict[str, set[str]] = {}
    for t in tasks:
        path = t.metadata.dataset.get("path", t.metadata.name)
        domains = set(getattr(t.metadata, "domains", None) or [])
        dataset_domains.setdefault(path, set()).update(domains)

    dataset_counter = Counter()
    for path, domains in dataset_domains.items():
        for d in domains:
            dataset_counter[d] += 1
    n_datasets = len(dataset_domains)

    print(
        f"\nTask-level domain counts ({len(tasks)} tasks, multi-domain tasks counted in each):"
    )
    for d, c in task_counter.most_common():
        print(f"  {d:18s} {c}")

    print(f"\nDataset-level domain counts ({n_datasets} unique datasets):")
    for d, c in dataset_counter.most_common():
        print(f"  {d:18s} {c}")

    # Plot dataset-level counts (more honest re: diversity)
    domains = [d for d, _ in dataset_counter.most_common()]
    counts = [dataset_counter[d] for d in domains]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bars = ax.barh(
        domains[::-1], counts[::-1], color="#4C72B0", edgecolor="black", linewidth=0.5
    )
    ax.set_xlabel(f"Number of source datasets (of {n_datasets} total)")
    ax.set_ylabel("Domain")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, count in zip(bars, counts[::-1]):
        ax.text(
            count + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_pdf = fig_dir / "domain_distribution.pdf"
    out_png = fig_dir / "domain_distribution.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory the figures are written to (default: this script's "
        "folder). Point at the paper's figures/ directory to write there directly.",
    )
    args = parser.parse_args()
    main(fig_dir=Path(args.out_dir))
