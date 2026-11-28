import pytest

import mteb

pytestmark = pytest.mark.core

_BENCHMARK = "MTEB(ara, v1)"
_ALLOWED_LANGUAGES = {"ara", "arb", "eng"}


def test_arabic_benchmark_alias():
    assert mteb.get_benchmark("MTEB(ara)").name == _BENCHMARK


def test_arabic_benchmark_only_arabic_and_english():
    """Bitext/cross-lingual subsets must not pull in other languages (see #2135)."""
    benchmark = mteb.get_benchmark(_BENCHMARK)
    assert benchmark.tasks.languages <= _ALLOWED_LANGUAGES


def test_arabic_benchmark_no_machine_translated_or_lm_generated_tasks():
    benchmark = mteb.get_benchmark(_BENCHMARK)
    for task in benchmark.tasks:
        creation = task.metadata.sample_creation or ""
        annotators = task.metadata.annotations_creators or ""
        assert "machine-translated" not in creation, task.metadata.name
        assert "LM-generated" not in creation, task.metadata.name
        assert "LM-generated" not in annotators, task.metadata.name


def test_arabic_benchmark_covers_task_types():
    benchmark = mteb.get_benchmark(_BENCHMARK)
    types = {t.metadata.type for t in benchmark.tasks}
    assert types >= {
        "Retrieval",
        "Reranking",
        "STS",
        "PairClassification",
        "Classification",
        "Clustering",
        "BitextMining",
    }
