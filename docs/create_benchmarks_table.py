from __future__ import annotations

from collections import Counter
from pathlib import Path

import mteb
from docs.create_tasks_table import insert_tables


def benchmark_to_markdown_row(b: mteb.Benchmark) -> str:
    name = b.name
    name_w_reference = f"[{name}]({b.reference})" if b.reference else name
    num_tasks = len(b.tasks)
    n_tasks = f"{num_tasks}"

    agg_domains = set()
    agg_langs = set()
    for t in b.tasks:
        if t.metadata.domains:
            agg_domains.update(t.metadata.domains)
        if t.metadata.languages:
            agg_langs.update(t.languages)

    langs = ",".join(list(agg_langs))
    domains = "[" + ", ".join(agg_domains) + "]" if agg_domains else ""

    task_types = dict(Counter([t.metadata.type for t in b.tasks]))

    return f"| {name_w_reference} | {n_tasks} | {task_types} | {domains} | {langs} |"


def create_benchmarks_table(benchmarks: list[mteb.Benchmark]) -> str:
    table = """
| Name | # Tasks | Task Types | Domains | Languages |
|------|---------|------------|---------|-----------|
"""
    for benchmark in benchmarks:
        table += benchmark_to_markdown_row(benchmark) + "\n"
    return table


def main():
    benchmarks = mteb.get_benchmarks()
    benchmarks = sorted(benchmarks, key=lambda x: x.name)

    benchmarks_table = create_benchmarks_table(benchmarks)

    file_path = Path(__file__).parent / "benchmarks.md"

    insert_tables(
        file_path,
        tables=[benchmarks_table],
        tags=["BENCHMARKS TABLE"],
    )


if __name__ == "__main__":
    main()
