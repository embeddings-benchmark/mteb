import bibtexparser
import pytest
from bibtexparser.writer import BibtexFormat

import mteb
from mteb.abstasks import AbsTask
from mteb.benchmarks.benchmark import Benchmark


def format_bibtex(bibtex_str: str) -> str | None:
    library = bibtexparser.parse_string(bibtex_str)
    if not library.entries:
        return None

    bib_format = BibtexFormat()
    bib_format.indent = "  "
    bib_format.trailing_comma = True
    bib_format.block_separator = "\n"

    return bibtexparser.write_string(library, bibtex_format=bib_format).strip()


@pytest.fixture(params=mteb.get_tasks())
def task(request: pytest.FixtureRequest):
    return request.param


def test_task_bibtex(task: AbsTask):
    task_name = task.metadata.name
    bibtex_citation = task.metadata.bibtex_citation

    if not bibtex_citation or not bibtex_citation.strip():
        pytest.skip(f"Task {task_name} has no bibtex_citation")
    bibtex_citation = bibtex_citation.strip()

    formatted_bibtex = format_bibtex(bibtex_citation)
    assert formatted_bibtex is not None and formatted_bibtex == bibtex_citation, (
        f"Wrong BibTeX citation formatting for task {task_name}"
    )


@pytest.fixture(params=mteb.get_benchmarks())
def benchmark(request: pytest.FixtureRequest):
    return request.param


def test_benchmark_bibtex(benchmark: Benchmark):
    benchmark_name = benchmark.name
    bibtex_citation = benchmark.citation

    if not bibtex_citation or not bibtex_citation.strip():
        pytest.skip(f"Benchmark {benchmark_name} has no bibtex_citation")
    bibtex_citation = bibtex_citation.strip()

    formatted_bibtex = format_bibtex(bibtex_citation)
    assert formatted_bibtex is not None and formatted_bibtex == bibtex_citation, (
        f"Wrong BibTeX citation formatting for benchmark {benchmark_name}"
    )
