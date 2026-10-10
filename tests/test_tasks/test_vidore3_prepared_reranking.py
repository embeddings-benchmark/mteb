"""Existing ViDoRe tasks support ordinary retrieval and prepared candidates."""

import json
from pathlib import Path
from typing import Any

import pytest

import mteb
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


@pytest.mark.parametrize("domain", _DOMAINS)
def test_existing_task_declares_prepared_sources(domain):
    task = mteb.get_task(f"Vidore3{domain}Retrieval.v2", hf_subsets=["english"])
    assert task.metadata.type == "DocumentUnderstanding"
    assert task.metadata.prompt == {
        "query": "Find a screenshot that is relevant to the user's question."
    }
    assert task.hf_subsets == ["english"]
    assert all(
        Path(source.filename).name == task.prediction_file_name
        for source in task.first_stage_predictions.values()
    )
    assert set(task.first_stage_predictions) == {
        "bm25-text",
        "bge-text",
        "qwen-text-image",
    }
    assert all(
        source.revision == "3d6834bc0d3aded9de65eb2e431d875f654c96e8"
        for source in task.first_stage_predictions.values()
    )
    assert task.metadata.descriptive_stats is not None
    assert task._reranking_experiment is None
    assert not task.data_loaded


@pytest.mark.parametrize(
    ("source", "expected"),
    [("bm25-text", "d1"), ("bge-text", "d2"), ("qwen-text-image", "d2")],
)
def test_retrieval_and_reranking_keep_task_identity(
    tmp_path, monkeypatch, source, expected
):
    task = mteb.get_task("Vidore3HrRetrieval.v2", hf_subsets=["english"])
    before = task.metadata.model_dump()
    mock = MockRetrievalTask()
    mock.load_data()
    shared = mock.dataset["default"]["test"]

    def load_data(self, **kwargs: Any):
        self.dataset = {"english": {"test": shared}}
        self.data_loaded = True

    monkeypatch.setattr(type(task), "load_data", load_data)
    monkeypatch.setattr(
        "mteb.abstasks.first_stage_predictions.hf_hub_download",
        lambda **kwargs: str(tmp_path / kwargs["filename"]),
    )
    for name, declaration in task.first_stage_predictions.items():
        path = tmp_path / declaration.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        ranking = {"d1": 2, "d2": 1} if name == "bm25-text" else {"d2": 2, "d1": 1}
        path.write_text(
            json.dumps(
                {
                    "mteb_model_meta": {"model_name": name},
                    "english": {"test": {"q1": ranking, "q2": ranking}},
                }
            )
        )
    model = mteb.get_model("mteb/baseline-random-encoder")
    cache = mteb.ResultCache(tmp_path / "cache")
    retrieval = mteb.evaluate(model, task, cache=cache, co2_tracker=False)
    assert retrieval.experiment_name is None
    qrels = shared["relevant_docs"]
    task.convert_to_reranking(first_stage=source, top_k=1)
    assert shared["top_ranked"] == {"q1": [expected], "q2": [expected]}
    assert shared["relevant_docs"] is qrels
    assert task._reranking_experiment["name"] == source
    assert (
        task._reranking_experiment["predictions"]["repo_id"]
        == "mteb/Vidore3RetrievalPredictions"
    )
    reranking = mteb.evaluate(model, task, cache=cache, co2_tracker=False)
    assert reranking[0].task_name == retrieval[0].task_name == "Vidore3HrRetrieval.v2"
    assert task.metadata.model_dump() == before
    assert (
        reranking.model_meta.experiment_kwargs["first_stage"]
        == task._reranking_experiment
    )
    rows = cache.load_results(
        models=[retrieval.model_meta, reranking.model_meta],
        tasks=[task],
        include_remote=False,
    )._to_dataset()
    assert len(rows) == 2
    assert {row["experiments"] is None for row in rows} == {True, False}
