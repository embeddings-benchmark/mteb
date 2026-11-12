import difflib
import re
from collections import defaultdict
from itertools import combinations

import pytest

import mteb
from mteb import get_task, get_tasks
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskType
from mteb.filter_tasks import filter_tasks
from mteb.get_tasks import MTEBTasks
from mteb.types import Modalities

pytestmark = pytest.mark.core


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


def test_get_tasks_warning_with_beta_tasks():
    """Test that get_tasks gives a warning when beta tasks are included"""
    with pytest.warns(UserWarning, match="is currently in beta"):
        get_tasks(exclude_beta=False)


@pytest.mark.parametrize("language", ["zho-Hant", "srp-Latn"])
def test_get_tasks_language_with_script_filter_regression_5720(language: str):
    """Raised in https://github.com/embeddings-benchmark/mteb/pull/5720/"""
    tasks = get_tasks(languages=[language])

    assert tasks
    for task in tasks:
        assert language in task.bcp47_codes, (
            f"Task {task.metadata.name} does not contain the correct language script"
        )


def test_get_tasks_language_and_separate_script_filter_regression_5720():
    """Raised in https://github.com/embeddings-benchmark/mteb/pull/5720/ (control: a bare language with a separate script)"""
    tasks = get_tasks(languages=["zho"], script=["Hant"])

    assert tasks
    for task in tasks:
        assert "zho-Hant" in task.bcp47_codes


@pytest.mark.parametrize(
    ("languages", "script"),
    [(["srp-Latn"], ["Latn"]), (["srp-Latn"], ["Cyrl"])],
)
def test_language_script_code_with_script_filter_raises_regression_5720(
    languages, script
):
    """Raised in https://github.com/embeddings-benchmark/mteb/pull/5720/"""
    with pytest.raises(ValueError, match="script"):
        get_tasks(languages=languages, script=script)
    with pytest.raises(ValueError, match="script"):
        get_tasks(tasks=["FloresBitextMining"], languages=languages, script=script)
    with pytest.raises(ValueError, match="script"):
        filter_tasks(get_tasks(), languages=languages, script=script)
    with pytest.raises(ValueError, match="script"):
        get_task("FloresBitextMining").filter_languages(languages, script)


def test_get_tasks_language_and_script_regression_5768():
    """Raised in https://github.com/embeddings-benchmark/mteb/issues/5768

    Without `tasks=`, a language and script filter raised if a task had the language in one
    subset and the script in another (e.g. MultiEURLEX has `eng-Latn` and `bul-Cyrl`, but no
    English or Russian in Cyrillic). Such a task should be skipped, as it is with `tasks=`.
    """
    kwargs = dict(languages=["eng", "rus"], script=["Cyrl"])

    tasks = get_tasks(**kwargs)

    assert tasks
    for task in tasks:
        assert {"eng-Cyrl", "rus-Cyrl"} & set(task.bcp47_codes)

    names = {t.metadata.name for t in tasks}
    assert "GeoreviewClassification.v2" in names
    assert "MultiEURLEXMultilabelClassification.v2" not in names
    with_tasks = get_tasks(
        tasks=["GeoreviewClassification.v2", "MultiEURLEXMultilabelClassification.v2"],
        **kwargs,
    )
    assert [t.metadata.name for t in with_tasks] == ["GeoreviewClassification.v2"]


def test_get_tasks_exclusive_language_filter():
    """Test that exclusive_language_filter removes subsets that contain other languages"""
    [sts22] = get_tasks(tasks=["STS22.v2"], languages=["eng"])
    assert sts22.hf_subsets == ["en", "de-en", "es-en", "pl-en", "zh-en"]

    [sts22] = get_tasks(
        tasks=["STS22.v2"], languages=["eng"], exclusive_language_filter=True
    )
    assert sts22.hf_subsets == ["en"]


def test_get_tasks_exclusive_language_filter_without_tasks_regression_5768():
    """Raised in https://github.com/embeddings-benchmark/mteb/issues/5768

    `exclusive_language_filter` was only applied with `tasks=`, so without it cross-lingual
    subsets such as `de-en` were kept for an English-only filter.
    """
    tasks = get_tasks(
        languages=["eng"], task_types=["STS"], exclusive_language_filter=True
    )

    sts22 = next(t for t in tasks if t.metadata.name == "STS22.v2")
    assert sts22.hf_subsets == ["en"]


def test_get_tasks_exclusive_language_filter_without_languages_warns():
    """Raised in https://github.com/embeddings-benchmark/mteb/pull/5826

    `exclusive_language_filter` needs `languages`; without them it is ignored with a warning
    instead of removing every task.
    """
    with pytest.warns(UserWarning, match="exclusive_language_filter"):
        tasks = get_tasks(exclusive_language_filter=True)

    assert tasks
    assert [t.metadata.name for t in tasks] == [t.metadata.name for t in get_tasks()]
