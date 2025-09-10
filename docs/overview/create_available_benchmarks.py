"""Updates the available benchmarks markdown file."""

from __future__ import annotations

from pathlib import Path

import mteb

START_INSERT = "<!-- START TASK DESCRIPTION -->"
END_INSERT = "<!-- END TASK DESCRIPTION -->"

benchmark_entry = """
####  {benchmark_name}

{description}

??? info Tasks

{task_table}
"""
learn_more = "[Learn more →]({reference})"


def format_benchmark_entry(benchmark: mteb.Benchmark) -> str:
    description = benchmark.description or "No description available."
    if benchmark.reference:
        # add learn more link to description
        description += (
            "\n\n" + learn_more.format(reference=benchmark.reference) + "\n\n"
        )

    tasks_md = benchmark.tasks.to_markdown(["type", "modalities"])

    task_md_indented = "\n".join([f"    {line}" for line in tasks_md.split("\n")])

    return benchmark_entry.format(
        benchmark_name=benchmark.name,
        description=description,
        task_table=task_md_indented,
    )


def insert_between_markers(
    content: str,
    insert: str,
    start_marker: str = START_INSERT,
    end_marker: str = END_INSERT,
) -> str:
    """Insert `insert` between the `start_marker` and `end_marker` in `content`. Delete any content in between.

    Keeps the markers.
    """
    start_idx = content.index(start_marker) + len(start_marker)
    end_idx = content.index(end_marker)
    new_content = content[:start_idx] + "\n" + insert + "\n" + content[end_idx:]
    return new_content


def main(path: Path) -> None:
    benchmarks = mteb.get_benchmarks()

    benchmark_entries = ""
    for benchmark in sorted(benchmarks, key=lambda b: b.name):
        benchmark_entries += format_benchmark_entry(benchmark) + "\n"

    with path.open("r") as f:
        content = f.read()
    new_content = insert_between_markers(content, benchmark_entries.strip())
    with path.open("w") as f:
        f.write(new_content)


if __name__ == "__main__":
    root = Path(__file__).parent / ".." / ".."
    main(root / "docs" / "overview" / "available_benchmarks.md")
