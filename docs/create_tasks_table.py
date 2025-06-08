from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import polars as pl

import mteb
from mteb.abstasks.TaskMetadata import TASK_TYPE
from mteb.languages import ISO_TO_FAM_LEVEL0, ISO_TO_LANGUAGE, PROGRAMMING_LANGS


def author_from_bibtex(bibtex: str | None) -> str:
    """Create (Authors, Year) from bibtex entry (author = {Authors}, year = {Year})"""
    if bibtex is None:
        return ""
    # get authors from bibtex (author = {Authors} or author={Authors})
    authors = re.search(r"author\s*=\s*{([^}]*)}", bibtex)
    if authors is None:
        return ""
    authors = authors.group(1)
    authors = [a.split(", ") for a in authors.split(" and ")]
    author_str_w_et_al = (
        authors[0][0] + " et al." if len(authors[0]) > 1 else authors[0][0]
    )
    # replace any newline characters
    author_str_w_et_al = author_str_w_et_al.replace("\n", " ")
    year = re.search(r"year\s*=\s*{([^}]*)}", bibtex)
    if year is None:
        return ""
    year_str = year.group(1)
    return f" ({author_str_w_et_al}, {year_str})"


def round_floats_in_dict(d: dict, precision: int = 2) -> dict:
    if not isinstance(d, dict):
        return d
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = round(value, precision)
        elif isinstance(value, dict):
            d[key] = round_floats_in_dict(value, precision)
    return d


def task_to_markdown_row(task: mteb.AbsTask) -> str:
    name = task.metadata.name
    name_w_reference = (
        f"[{name}]({task.metadata.reference})" if task.metadata.reference else name
    )
    domains = (
        "[" + ", ".join(sorted(task.metadata.domains)) + "]"
        if task.metadata.domains
        else ""
    )
    n_samples = task.metadata.n_samples
    dataset_statistics = round_floats_in_dict(task.metadata.descriptive_stats)
    name_w_reference += author_from_bibtex(task.metadata.bibtex_citation)

    return f"| {name_w_reference} | {task.metadata.languages} | {task.metadata.type} | {task.metadata.category} | {domains} | {n_samples} | {dataset_statistics} |"


def create_tasks_table(tasks: list[mteb.AbsTask]) -> str:
    table = """
| Name | Languages | Type | Category | Domains | # Samples | Dataset statistics |
|------|-----------|------|----------|---------|-----------|--------------------|
"""
    for task in tasks:
        table += task_to_markdown_row(task) + "\n"
    return table


def create_task_lang_table(tasks: list[mteb.AbsTask], sort_by_sum=False) -> str:
    table_dict = {}
    ## Group by language. If it is a multilingual dataset, 1 is added to all languages present.
    for task in tasks:
        for lang in task.metadata.languages:
            if lang in PROGRAMMING_LANGS:
                lang = "code"
            if table_dict.get(lang) is None:
                table_dict[lang] = {k: 0 for k in sorted(get_args(TASK_TYPE))}
            table_dict[lang][task.metadata.type] += 1

    ## Wrangle for polars
    pl_table_dict = []
    for lang, d in table_dict.items():
        d.update({"0-lang-code": lang})  # for sorting columns
        pl_table_dict.append(d)

    df = pl.DataFrame(pl_table_dict).sort(by="0-lang-code")
    df = df.with_columns(
        pl.col("0-lang-code")
        .replace_strict(ISO_TO_LANGUAGE, default="unknown")
        .alias("1-lang-name")
    )
    df = df.with_columns(
        pl.col("0-lang-code")
        .replace_strict(ISO_TO_FAM_LEVEL0, default="Unclassified")
        .alias("2-lang-fam")
    )

    df = df.with_columns(sum=pl.sum_horizontal(get_args(TASK_TYPE)))
    df = df.select(sorted(df.columns))
    if sort_by_sum:
        df = df.sort(by="sum", descending=True)

    total = df.sum()
    task_names = sorted(get_args(TASK_TYPE))
    headers = ["ISO Code", "Language", "Family"] + task_names + ["Sum"]
    table_header = "| " + " | ".join(headers) + " |"
    separator_line = "|"
    for header in headers:
        width = len(header) + 2
        separator_line += "-" * width + "|"
    table = table_header + "\n" + separator_line + "\n"

    for row in df.iter_rows():
        table += f"| {row[0]} "
        for num in row[1:]:
            table += f"| {num} "
        table += "|\n"

    for row in total.iter_rows():
        table += "| Total "
        for num in row[:-1]:
            table += f"| {num} "
        table += "|\n"

    return table


def insert_tables(
    file_path: str, tables: list[str], tags: list[str] = ["TASKS TABLE"]
) -> None:
    """Insert tables within <!-- TABLE START --> and <!-- TABLE END --> or similar tags."""
    md = Path(file_path).read_text(encoding="utf-8")

    for table, tag in zip(tables, tags):
        start = f"<!-- {tag} START -->"
        end = f"<!-- {tag} END -->"
        # Ensure a newline after the start tag
        md = md.replace(
            md[md.index(start) + len(start) : md.index(end)],
            f"\n{table}\n",
        )

    Path(file_path).write_text(md, encoding="utf-8")


def main():
    tasks = mteb.get_tasks()
    tasks = sorted(tasks, key=lambda x: x.metadata.name)

    tasks_table = create_tasks_table(tasks)
    task_lang_table = create_task_lang_table(tasks)

    file_path = Path(__file__).parent / "tasks.md"

    insert_tables(
        file_path,
        tables=[tasks_table, task_lang_table],
        tags=["TASKS TABLE", "TASK LANG TABLE"],
    )


if __name__ == "__main__":
    main()
