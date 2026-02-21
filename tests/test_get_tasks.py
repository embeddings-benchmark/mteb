import difflib
import re
from collections import defaultdict
from itertools import combinations

import pytest

import mteb
from mteb import get_task, get_tasks
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskType
from mteb.get_tasks import MTEBTasks
from mteb.types import Modalities


@pytest.mark.parametrize(
    "task_name", ["BornholmBitextMining", "CQADupstackRetrieval", "Birdsnap"]
)
@pytest.mark.parametrize("eval_splits", [["test"], None])
def test_get_task(
    task_name: str,
    eval_splits: list[str] | None,
):
    task = get_task(
        task_name,
        eval_splits=eval_splits,
    )
    assert isinstance(task, AbsTask)
    assert task.metadata.name == task_name
    if eval_splits:
        for split in task.eval_splits:
            assert split in eval_splits
    else:
        assert task.eval_splits == task.metadata.eval_splits


def test_get_tasks_filtering():
    """Tests that get_tasks filters tasks for languages within the task, i.e. that a multilingual task returns only relevant subtasks for the
    specified languages
    """
    tasks = get_tasks(languages=["eng"])

    for task in tasks:
        if (
            task.metadata.is_multilingual
            and task.metadata.name != "STS17MultilingualVisualSTSEng"
        ):
            assert isinstance(task.metadata.eval_langs, dict)

            for hf_subset in task.hf_subsets:
                assert "eng-Latn" in task.metadata.eval_langs[hf_subset], (
                    f"{task.metadata.name}"
                )


@pytest.mark.parametrize("script", [["Cyrl"], None])
@pytest.mark.parametrize("task_types", [["Classification"], ["Clustering"], None])
@pytest.mark.parametrize("modalities", [["text"], ["image"], None])
def test_mteb_mteb_tasks(
    script: list[str],
    task_types: list[TaskType] | None,
    modalities: list[Modalities] | None,
):
    tasks = mteb.get_tasks(script=script, task_types=task_types, modalities=modalities)
    assert isinstance(tasks, MTEBTasks)
    langs = tasks.languages
    for t in tasks:
        assert len(langs.intersection(t.languages)) > 0

    # check for header of a table
    n_langs = len(tasks)
    assert len(tasks.to_markdown().split("\n")) - 3 == n_langs


@pytest.mark.parametrize("modalities", [["text"], ["image"], ["text", "image"]])
def test_get_tasks_with_exclusive_modality_filter(modalities):
    """Test exclusive_modality_filter with actual tasks (if available)"""
    text_tasks_exclusive = get_tasks(
        modalities=modalities, exclusive_modality_filter=True
    )
    for task in text_tasks_exclusive:
        assert set(task.modalities) == set(modalities)


def test_get_tasks_privacy_filtering():
    """Test that get_tasks correctly filters by privacy status"""
    # By default, should only return public datasets (exclude_private=True)
    public_tasks = get_tasks()

    # Should include private datasets when explicitly requested
    all_tasks = get_tasks(exclude_private=False)

    # All tasks should contain at least as many or more tasks than public tasks
    assert len(all_tasks) >= len(public_tasks)

    # All returned tasks should be public when exclude_private=True
    for task in public_tasks:
        assert (
            task.metadata.is_public is not False
        )  # None or True are both considered public


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
    return " ".join(title.lower().strip().split())


def _get_duplicate_citations() -> list[tuple[str, str, str, str, str, str]]:
    """Same paper under different bibtex ids -> (task1, task2, id1, id2, raw_title_1, raw_title_2)."""
    by_title: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for task_cls in mteb.get_tasks():
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
    for norm_title, items in by_title.items():
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
        if items[0][2] in (
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
        ):
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
