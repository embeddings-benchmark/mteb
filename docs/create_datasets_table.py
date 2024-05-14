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


def dataset_to_markdown_row(dataset: mteb.Absdataset) -> str:
    name = dataset.metadata.name
    name_w_reference = (
        f"[{name}]({dataset.metadata.reference})" if dataset.metadata.reference else name
    )
    domains = (
        "[" + ", ".join(dataset.metadata.domains) + "]" if dataset.metadata.domains else ""
    )
    n_samples = dataset.metadata.n_samples if dataset.metadata.n_samples else ""
    avg_character_length = (
        dataset.metadata.avg_character_length if dataset.metadata.avg_character_length else ""
    )

    name_w_reference += author_from_bibtex(dataset.metadata.bibtex_citation)

    return f"| {name_w_reference} | {dataset.metadata.languages} | {dataset.metadata.type} | {dataset.metadata.category} | {domains} | {n_samples} | {avg_character_length} |"


def create_datasets_table(datasets: list[mteb.AbsTask]) -> str:
    table = """
| Name | Languages | Task | Category | Domains | # Samples | Avg. Length (Char.) |
|------|-----------|------|----------|---------|-----------|---------------------|
"""
    for dataset in datasets:
        table += dataset_to_markdown_row(dataset) + "\n"
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
    datasets = mteb.get_datasets()
    datasets = sorted(datasets, key=lambda x: x.metadata.name)

    table = create_datasets_table(datasets)

    file_path = Path(__file__).parent / "datasets.md"

    insert_table(file_path, table)


if __name__ == "__main__":
    main()
