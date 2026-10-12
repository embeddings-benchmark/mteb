"""Prepared first stages behave like ordinary, independently selectable subsets."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import mteb
from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource
from mteb.benchmarks._benchmark_metrics import _compute_custom_group_means
from mteb.benchmarks.benchmark import CustomGroup, CustomGrouping
from mteb.mocks.mock_tasks import MockRetrievalTask

_DOMAINS = (
    "ComputerScience",
    "Energy",
    "FinanceEn",
    "FinanceFr",
    "Hr",
    "Industrial",
    "Pharmaceuticals",
    "Physics",
)
_SOURCES = ("bm25-text", "bge-text", "qwen-text-image")


@pytest.mark.parametrize("domain", _DOMAINS)
def test_registered_task_defaults_and_language_selection(domain):
    task = mteb.get_task(f"Vidore3{domain}Reranking")
    retrieval = mteb.get_task(f"Vidore3{domain}Retrieval.v2")
    assert task.metadata.type == "Reranking"
    assert task.metadata.dataset == retrieval.metadata.dataset
    assert task.metadata.adapted_from == [retrieval.metadata.name]
    assert len(task.hf_subsets) == 18
    assert task.top_k == 50
    assert not task.data_loaded
    english = mteb.get_task(task.metadata.name, languages=["eng"])
    assert english.hf_subsets == [f"{source}__english" for source in _SOURCES]
    for source in task.first_stage_predictions.values():
        assert Path(source.filename).name == retrieval.prediction_file_name
        assert source.revision == "3d6834bc0d3aded9de65eb2e431d875f654c96e8"
    with pytest.raises(ValueError, match="No subsets"):
        mteb.get_task(task.metadata.name, hf_subsets=["unknown__english"])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    task = mteb.get_task("Vidore3HrReranking")
    mock = MockRetrievalTask()
    mock.load_data()
    shared = mock.dataset["default"]["test"]
    original_qrels = deepcopy(shared["relevant_docs"])
    loads, downloads = [], []

    def load_data(self, **kwargs: Any):
        loads.append(list(self.hf_subsets))
        self.dataset = {subset: {"test": dict(shared)} for subset in self.hf_subsets}
        self.data_loaded = True

    def download(**kwargs: Any):
        downloads.append(kwargs["filename"])
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(task.retrieval_task, "load_data", load_data)
    monkeypatch.setattr(
        "mteb.abstasks.first_stage_predictions.hf_hub_download", download
    )
    for name, source in task.first_stage_predictions.items():
        ranking = (
            {"d1": 2.0, "d2": 1.0} if name == "bm25-text" else {"d2": 2.0, "d1": 1.0}
        )
        data = {
            "mteb_model_meta": {"model_name": name, "revision": "abc"},
            **{
                language: {"test": {"q1": ranking, "q2": ranking}}
                for language in task.retrieval_task.metadata.hf_subsets
            },
        }
        path = tmp_path / source.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return task, shared, original_qrels, loads, downloads


def test_default_load_shares_data_and_preserves_each_candidate_set(prepared):
    task, shared, qrels, loads, downloads = prepared
    task.load_data()
    task.load_data()
    assert len(loads) == 1 and len(loads[0]) == 6
    assert len(downloads) == 3
    assert len(task.dataset) == 18
    for subset, splits in task.dataset.items():
        data = splits["test"]
        assert data is not shared
        assert data["corpus"] is shared["corpus"]
        assert data["queries"] is shared["queries"]
        assert data["relevant_docs"] == qrels
        expected = ["d1", "d2"] if subset.startswith("bm25") else ["d2", "d1"]
        assert data["top_ranked"]["q1"] == expected
    assert shared["top_ranked"] is None


def test_filter_loads_only_requested_sources_and_languages(prepared):
    _, _, _, loads, downloads = prepared
    task = mteb.get_task("Vidore3HrReranking", hf_subsets=["bge-text__english"])
    task.load_data()
    assert loads == [["english"]]
    assert len(downloads) == 1 and "BAAI__bge-m3" in downloads[0]
    assert list(task.dataset) == ["bge-text__english"]


def test_one_result_file_resumes_subsets_and_exports_retriever_groups(  # noqa: PLR0914
    prepared, tmp_path, monkeypatch
):
    task, _, _, _, _ = prepared
    model = mteb.get_model("mteb/baseline-random-encoder")
    cache = mteb.ResultCache(tmp_path / "cache")
    prediction_dir = tmp_path / "predictions"
    calls = []
    original = type(task)._evaluate_subset

    def evaluate_subset(self, *args: Any, hf_subset: str, **kwargs: Any):
        calls.append(hf_subset)
        return original(self, *args, hf_subset=hf_subset, **kwargs)

    monkeypatch.setattr(type(task), "_evaluate_subset", evaluate_subset)
    one = mteb.get_task(task.metadata.name, hf_subsets=["bge-text__english"])
    mteb.evaluate(
        model, one, cache=cache, co2_tracker=False, prediction_folder=prediction_dir
    )
    all_english = mteb.get_task(task.metadata.name, languages=["eng"])
    result = mteb.evaluate(
        model,
        all_english,
        cache=cache,
        co2_tracker=False,
        prediction_folder=prediction_dir,
    )
    assert sorted(calls) == sorted(f"{source}__english" for source in _SOURCES)
    assert result.experiment_name is None
    assert model.mteb_model_meta.experiment_kwargs is None
    assert len(result[0].scores["test"]) == 3
    for score in result[0].scores["test"]:
        source = score["hf_subset"].split("__")[0]
        assert score["previous_results_model_meta"]["model_name"] == source
        assert score["languages"] == ["eng-Latn"]
    files = list((tmp_path / "cache").rglob(f"{task.metadata.name}.json"))
    assert len(files) == 1 and "experiments" not in files[0].parts
    cached = mteb.evaluate(
        model,
        all_english,
        cache=cache,
        co2_tracker=False,
        overwrite_strategy="only-cache",
    )
    assert len(calls) == 3
    assert cached[0].get_score() == result[0].get_score()
    predictions = json.loads((prediction_dir / task.prediction_file_name).read_text())
    assert all(f"{source}__english" in predictions for source in _SOURCES)
    loaded = cache.load_results(
        models=[model.mteb_model_meta], tasks=[all_english], include_remote=False
    )
    rows = loaded._to_dataset()
    assert len(rows) == 3
    assert all(row["experiments"] is None for row in rows)
    grouping = CustomGrouping(
        name="First-stage retriever",
        groups=tuple(
            CustomGroup(
                label=source,
                tasks=[
                    mteb.get_task(task.metadata.name, hf_subsets=[f"{source}__english"])
                ],
            )
            for source in _SOURCES
        ),
    )
    grouped = _compute_custom_group_means([result[0]], grouping)
    for score in result[0].scores["test"]:
        assert (
            grouped[f"First-stage retriever::{score['hf_subset'].split('__')[0]}"]
            == score["main_score"]
        )


def test_pinned_source_requires_immutable_revision():
    with pytest.raises(ValueError, match="commit SHA"):
        FirstStagePredictionSource("org/preds", "predictions.json", "main")
