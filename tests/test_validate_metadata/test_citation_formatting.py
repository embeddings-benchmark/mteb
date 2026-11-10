import difflib
import re
from collections import defaultdict
from itertools import combinations

import bibtexparser
import pytest
from bibtexparser.writer import BibtexFormat

import mteb
from mteb.abstasks import AbsTask
from mteb.benchmarks.benchmark import Benchmark

pytestmark = pytest.mark.core


def format_bibtex(bibtex_str: str) -> str | None:
    library = bibtexparser.parse_string(bibtex_str)
    if not library.entries:
        return None
    for comment in list(library.comments):
        library.remove(comment)

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


def test_same_citation_id_has_identical_contents():
    """A shared BibTeX ID must have the same entry type and fields everywhere."""
    citations = [
        (f"task:{task.metadata.name}", task.metadata.bibtex_citation)
        for task in mteb.get_tasks(
            exclude_beta=False, exclude_superseded=False, exclude_private=False
        )
    ]
    citations.extend(
        (f"model:{model.name}", model.citation) for model in mteb.get_model_metas()
    )
    citations.extend(
        (f"benchmark:{benchmark.name}", benchmark.citation)
        for benchmark in mteb.get_benchmarks()
    )

    seen: dict[str, tuple[str, tuple[str, dict[str, str]]]] = {}
    conflicts = []
    for source, citation in citations:
        if not citation or not citation.strip():
            continue
        library = bibtexparser.parse_string(citation)
        assert not library.failed_blocks, f"Invalid BibTeX in {source}"
        for entry in library.entries:
            contents = (
                entry.entry_type.lower(),
                {
                    name.lower(): re.sub(r"\s+", " ", field.value).strip()
                    for name, field in entry.fields_dict.items()
                },
            )
            if entry.key not in seen:
                seen[entry.key] = (source, contents)
                continue
            previous_source, previous_contents = seen[entry.key]
            if contents != previous_contents:
                conflicts.append(f"{entry.key!r}: {previous_source} / {source}")

    assert not conflicts, (
        "Found citation IDs with different contents. Unify their entry types and "
        "fields, or use different IDs for distinct papers.\n" + "\n".join(conflicts)
    )


def _parse_bibtex_entries(bibtex_str: str) -> list[tuple[str, str]]:
    if not bibtex_str or not bibtex_str.strip():
        return []
    entries: list[tuple[str, str]] = []
    block_pattern = re.compile(r"@\w+\s*\{([^,]+)", re.IGNORECASE)
    for key_match in block_pattern.finditer(bibtex_str):
        citation_id = key_match.group(1).strip()
        block_start = key_match.start()
        next_at = bibtex_str.find("@", key_match.end())
        block = bibtex_str[block_start : next_at if next_at != -1 else None]
        title_match = re.search(r"\btitle\s*=\s*\{", block, re.IGNORECASE)
        if not title_match:
            continue
        start = title_match.end()
        depth = 1
        i = start
        while i < len(block) and depth > 0:
            if block[i] == "{":
                depth += 1
            elif block[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            title = block[start : i - 1].replace("\n", " ").strip()
            title = " ".join(title.split())
            if title:
                entries.append((citation_id, title))
    return entries


def _normalize_title_for_comparison(title: str) -> str:
    # braces only protect capitalisation, e.g. {BEIR}-{NL}
    title = title.replace("{", "").replace("}", "")
    return " ".join(title.lower().strip().split())


# different papers whose titles are close enough to look like duplicates
_DISTINCT_SIMILAR_TITLES = {
    _normalize_title_for_comparison(title)
    for title in (
        "BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language",
        "BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language",
        "Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition",
        "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition",
        "Overview of the TREC 2019 deep learning track",
        "Overview of the TREC 2020 deep learning track",
        "Overview of the TREC 2022 NeuCLIR track",
        "Overview of the TREC 2023 NeuCLIR Track",
        "MoD-Embedding: A Fine-tuned Multilingual Text Embedding Model",
        "Octen-Embedding-8B: A Fine-tuned Multilingual Text Embedding Model",
        "Qwen2.5-Omni Technical Report",
        "Qwen3-Omni Technical Report",
        "Granite Embedding Models",
        "Granite Embedding R2 Models",
    )
}


def _get_duplicate_citations() -> list[tuple[str, str, str, str, str, str]]:
    """Same paper under different bibtex ids -> (task1, task2, id1, id2, raw_title_1, raw_title_2)."""
    by_title: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for task_cls in mteb.get_tasks(
        exclude_beta=False, exclude_superseded=False, exclude_private=False
    ):
        if task_cls.metadata.bibtex_citation is None:
            continue
        for cid, title in _parse_bibtex_entries(task_cls.metadata.bibtex_citation):
            norm = _normalize_title_for_comparison(title)
            by_title[norm].append((f"task:{task_cls.metadata.name}", cid, title))

    for model_meta in mteb.get_model_metas():
        if model_meta.citation is None:
            continue
        for cid, title in _parse_bibtex_entries(model_meta.citation):
            norm = _normalize_title_for_comparison(title)
            by_title[norm].append((f"model:{model_meta.name}", cid, title))

    for benchmark in mteb.get_benchmarks():
        if benchmark.citation is None:
            continue
        for cid, title in _parse_bibtex_entries(benchmark.citation):
            norm = _normalize_title_for_comparison(title)
            by_title[norm].append((f"benchmark:{benchmark.name}", cid, title))

    duplicates = []
    for items in by_title.values():
        id_to_raw = {cid: raw for _, cid, raw in items}
        id_to_item = {cid: item for item, cid, raw in items}
        if len(id_to_raw) < 2:
            continue
        unique_ids = sorted(id_to_raw)
        for id1, id2 in combinations(unique_ids, 2):
            duplicates.append(
                (
                    id_to_item[id1],
                    id_to_item[id2],
                    id1,
                    id2,
                    id_to_raw[id1],
                    id_to_raw[id2],
                )
            )

    all_titles = set(by_title.keys())
    for title, items in by_title.items():
        if title in _DISTINCT_SIMILAR_TITLES:
            continue
        use_titles = all_titles - {title}
        close_matches = difflib.get_close_matches(title, use_titles, n=2, cutoff=0.9)
        if close_matches:
            close_match_items = by_title[close_matches[0]][0]
            for item in items:
                duplicates.append(
                    (
                        item[0],
                        close_match_items[0],
                        item[1],
                        close_match_items[1],
                        item[2],
                        close_match_items[2],
                    )
                )
    return duplicates


def test_no_duplicate_citations_with_different_ids():
    """Ensure no task citations refer to the same paper under different BibTeX IDs."""
    duplicates = _get_duplicate_citations()
    assert not duplicates, (
        f"Found {len(duplicates)} duplicate citations (same paper, different BibTeX IDs). "
        "Unify citation keys or titles so each paper is cited once.\n\n"
        + "\n\n".join(
            f"--- Duplicate {i}: {item1} / {item2} ---\n"
            f"  id1 = {id1!r} ({item1})\n      title: {title1}\n"
            f"  id2 = {id2!r} ({item2})\n      title: {title2}"
            for i, (item1, item2, id1, id2, title1, title2) in enumerate(duplicates, 1)
        )
    )
