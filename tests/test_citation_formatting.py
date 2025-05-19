from __future__ import annotations

import bibtexparser
import pytest
from bibtexparser.bwriter import BibTexWriter

import mteb
from mteb.abstasks import AbsTask
from mteb.benchmarks.benchmark import Benchmark


def format_bibtex(bibtex_str: str) -> str | None:
    parser = bibtexparser.bparser.BibTexParser(
        common_strings=True, ignore_nonstandard_types=False, interpolate_strings=False
    )

    bib_database = bibtexparser.loads(bibtex_str, parser)
    if not bib_database.entries:
        return None

    writer = BibTexWriter()
    writer.indent = "  "
    writer.comma_first = False
    writer.add_trailing_comma = True

    return writer.write(bib_database).strip()


@pytest.fixture(params=mteb.get_tasks())
def task(request):
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
def benchmark(request):
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
