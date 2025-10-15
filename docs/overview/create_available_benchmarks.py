"""Updates the available benchmarks markdown file."""

from pathlib import Path
from typing import cast

import mteb
from mteb.get_tasks import MTEBTasks

START_INSERT = "<!-- START TASK DESCRIPTION -->"
END_INSERT = "<!-- END TASK DESCRIPTION -->"

benchmark_entry = """
####  {benchmark_name}

{description}

??? info "Tasks"

{task_table}
"""
learn_more = "[Learn more →]({reference})"

citation_admonition = """

??? quote "Citation"

{citation_chunk}

"""

citation_chunk = """
```bibtex
{bibtex_citation}
```
"""


def pretty_long_list(items: list[str], max_items: int = 5) -> str:
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f", ... ({len(items)})"


def create_table(benchmark: mteb.Benchmark) -> str:
    """Create a markdown table of tasks in the benchmark."""
    tasks = benchmark.tasks
    tasks = cast(MTEBTasks, tasks)
    df = tasks.to_dataframe(["name", "type", "modalities", "languages"])

    # add links to task names:
    # format: http://127.0.0.1:8000/overview/available_tasks/retrieval/#treccovid
    df["name"] = df.apply(
        lambda row: f"[{row['name']}](./available_tasks/{row['type'].lower()}.md#{row['name'].lower()})",
        axis=1,
    )
    df["modalities"] = df["modalities"].apply(lambda x: pretty_long_list(x))
    df["languages"] = df["languages"].apply(lambda x: pretty_long_list(x))

    tasks_md = df.to_markdown(index=False)

    return tasks_md


def format_benchmark_entry(benchmark: mteb.Benchmark) -> str:
    description = benchmark.description or "No description available."
    if benchmark.reference:
        # add learn more link to description
        description += (
            "\n\n" + learn_more.format(reference=benchmark.reference) + "\n\n"
        )

    tasks_md = create_table(benchmark)
    task_md_indented = "\n".join([f"    {line}" for line in tasks_md.split("\n")])

    entry = benchmark_entry.format(
        benchmark_name=benchmark.name,
        description=description,
        task_table=task_md_indented,
    )
    if benchmark.citation:
        citation = citation_chunk.format(bibtex_citation=benchmark.citation)
        citation = "\n".join([f"    {line}" for line in citation.split("\n")])
        entry += citation_admonition.format(citation_chunk=citation)

    return entry


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
