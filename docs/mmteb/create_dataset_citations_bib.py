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


def main():
    tasks = mteb.get_tasks()
    tasks = sorted(tasks, key=lambda x: x.metadata.name)
    extract_bibtex_to_file(tasks)


if __name__ == "__main__":
    main()
