from __future__ import annotations

from pathlib import Path

import bibtexparser

import mteb


def extract_bibtex_to_file(tasks: list[mteb.AbsTask]) -> None:
    """Parse the task and extract bibtex.

    :param tasks:
        List of tasks.
    """
    titles = []
    bibtex = []
    for task in tasks:
        library = bibtexparser.parse_string(task.metadata.bibtex_citation)
        try:
            # extract titles to remove duplicate citations.
            title = library.entries[0].key
        except IndexError:
            continue

        if len(library.failed_blocks) > 0:
            print("Some blocks failed to parse.", task.metadata.bibtex_citation)

        if title not in titles:
            titles.append(title)
            bibtex.append(bibtexparser.write_string(library))

    file_path = Path(__file__).parent / "dataset_citations.bib"

    with open(file_path, "a") as file:
        file.truncate()
        for bib in bibtex:
            file.write(bib.strip() + "\n\n")


def create_citations_table(tasks: list[mteb.AbsTask]) -> str:
    """Create tex

    :param tasks:
        List of tasks.
    """
    table = """
\\onecolumn
\\setlength\\extrarowheight{7pt}
\\begin{longtable}{L{3.5cm}|L{3.0cm}L{1.4cm}L{1.4cm}L{1.4cm}L{1.4cm}L{1.0cm}L{1.0cm}}
\\toprule
\\textbf{Dataset} & \\textbf{N. Langs} & \\textbf{Type} & \\textbf{Category} & \\textbf{Domains} & \\textbf{N. Docs} & 
\\textbf{Avg. Length} \\\\ 
\\midrule
\\endhead \\\\"""
    for task in tasks:
        table += task_to_tex_row(task) + "\n \\hline \n"
    table += "\\end{longtable}"
    return table


def task_to_tex_row(task: mteb.AbsTask) -> str:
    """Generate a single tex row

    :param task:
        A single mteb tasks.
    """
    name = task.metadata.name
    domains = (
        "[" + ", ".join(task.metadata.domains) + "]" if task.metadata.domains else ""
    )
    n_samples = (
        f"{sum(task.metadata.n_samples.values()) / len(task.metadata.n_samples.keys()):.2f}"
        if task.metadata.n_samples
        else ""
    )

    avg_character_length = (
        "{:.2f}".format(
            sum(task.metadata.avg_character_length.values())
            / len(task.metadata.avg_character_length.keys())
        )
        if task.metadata.avg_character_length
        else ""
    )
    library = bibtexparser.parse_string(task.metadata.bibtex_citation)
    try:
        cite_key = library.entries[0].key
        cite_key = "\\cite{" + cite_key + "}"
    except IndexError:
        cite_key = ""
    lang = str(len(task.metadata.languages)) + " "
    lang += (
        str(task.metadata.languages[:11])[1:-1] + "..."
        if len(task.metadata.languages) > 11
        else str(task.metadata.languages)[1:-1]
    )
    lang = lang.replace("'", "")

    return f"{name}{cite_key} & {lang} & {task.metadata.type} & {task.metadata.category} & {domains[1:-1]} & {n_samples} & {avg_character_length} \\\\"


def main():
    tasks = mteb.get_tasks()
    tasks = sorted(tasks, key=lambda x: x.metadata.name)
    extract_bibtex_to_file(tasks)
    print(create_citations_table(tasks))


if __name__ == "__main__":
    main()
