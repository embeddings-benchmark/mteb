from __future__ import annotations

import re
from pathlib import Path

import mteb


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

    languages = sorted(list(task.metadata.languages))

    return f"| {name_w_reference} | {languages} | {task.metadata.type} | {task.metadata.category} | {domains} | {n_samples} | {avg_character_length} |"


def create_tasks_table(tasks: list[mteb.AbsTask]) -> str:
    table = """
| Name | Languages | Type | Category | Domains | # Samples | Avg. Length (Char.) |
|------|-----------|------|----------|---------|-----------|---------------------|
"""
    for task in tasks:
        table += task_to_markdown_row(task) + "\n"
    return table


def insert_table(file_path, table):
    """Insert table in the in <!-- TABLE START --> and <!-- TABLE END -->"""
    with open(file_path, "r") as file:
        md = file.read()

    start = "<!-- TABLE START -->"
    end = "<!-- TABLE END -->"

    md = md.replace(md[md.index(start) + len(start) : md.index(end)], table)

    with open(file_path, "w") as file:
        file.write(md)


def main():
    tasks = mteb.get_tasks()
    tasks = sorted(tasks, key=lambda x: x.metadata.name)

    table = create_tasks_table(tasks)

    file_path = Path(__file__).parent / "tasks.md"

    insert_table(file_path, table)


if __name__ == "__main__":
    main()
