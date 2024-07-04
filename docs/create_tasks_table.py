from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import polars as pl

import mteb
from mteb.abstasks.TaskMetadata import PROGRAMMING_LANGS, TASK_TYPE


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


def task_to_markdown_row(task: mteb.AbsTask) -> str:
    name = task.metadata.name
    name_w_reference = (
        f"[{name}]({task.metadata.reference})" if task.metadata.reference else name
    )
    domains = (
        "[" + ", ".join(task.metadata.domains) + "]" if task.metadata.domains else ""
    )
    n_samples = task.metadata.n_samples if task.metadata.n_samples else ""
    avg_character_length = (
        task.metadata.avg_character_length if task.metadata.avg_character_length else ""
    )

    name_w_reference += author_from_bibtex(task.metadata.bibtex_citation)

    return f"| {name_w_reference} | {task.metadata.languages} | {task.metadata.type} | {task.metadata.category} | {domains} | {n_samples} | {avg_character_length} |"


def create_tasks_table(tasks: list[mteb.AbsTask]) -> str:
    table = """
| Name | Languages | Type | Category | Domains | # Samples | Avg. Length (Char.) |
|------|-----------|------|----------|---------|-----------|---------------------|
"""
    for task in tasks:
        table += task_to_markdown_row(task) + "\n"
    return table


def create_task_lang_table(tasks: list[mteb.AbsTask]) -> str:
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
        d.update({"lang": lang})
        pl_table_dict.append(d)

    df = pl.DataFrame(pl_table_dict).sort(by="lang")
    total = df.sum()

    task_names_md = " | ".join(sorted(get_args(TASK_TYPE)))
    horizontal_line_md = "---|---" * len(sorted(get_args(TASK_TYPE)))
    table = """
| Language | {} |
|{}|
""".format(task_names_md, horizontal_line_md)

    for row in df.iter_rows():
        table += f"| {row[-1]} "
        for num in row[:-1]:
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
    md = Path(file_path).read_text()

    for table, tag in zip(tables, tags):
        start = f"<!-- {tag} START -->"
        end = f"<!-- {tag} END -->"
        md = md.replace(md[md.index(start) + len(start) : md.index(end)], table)

    Path(file_path).write_text(md)


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
