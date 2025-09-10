from __future__ import annotations

from pathlib import Path

import mteb

START_INSERT = "<!-- START TASK DESCRIPTION -->"
END_INSERT = "<!-- END TASK DESCRIPTION -->"

benchmark_entry = """
####  {benchmark_name}

{description}

[Learn more →]({reference})

??? info Tasks

    {task_table}
"""


def format_benchmark_entry(benchmark: mteb.Benchmark) -> str:
    description = benchmark.description or "No description available."
    assert benchmark.reference is not None
    reference = benchmark.reference

    tasks_md = benchmark.tasks.to_markdown(["type", "modalities"])

    return benchmark_entry.format(
        benchmark_name=benchmark.name,
        description=description,
        reference=reference,
        task_table=tasks_md,
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

    doc_benchmarks = path / "available_benchmarks.md"
    with doc_benchmarks.open("r") as f:
        content = f.read()
    new_content = insert_between_markers(content, benchmark_entries.strip())
    with doc_benchmarks.open("w") as f:
        f.write(new_content)


if __name__ == "__main__":
    root = Path(__file__).parent / ".." / ".."
    main(root / "docs" / "overview" / "available_benchmarks.md")
