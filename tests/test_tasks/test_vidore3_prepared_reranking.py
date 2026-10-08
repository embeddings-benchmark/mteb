"""Registered ViDoRe views reuse retrieval data and select prepared candidates."""

import json
from pathlib import Path

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
def test_registered_reranking_view_preserves_parent(domain):
    parent = mteb.get_task(f"Vidore3{domain}Retrieval.v2")
    before = parent.metadata.model_dump()
    task = mteb.get_task(f"Vidore3{domain}Reranking", hf_subsets=["english"])
    assert isinstance(task, type(parent))
    assert task.metadata.dataset == parent.metadata.dataset
    assert task.metadata.type == "Reranking"
    assert task.metadata.prompt == {
        "query": "Retrieve images or text relevant to the user's query."
    }
    assert task.metadata.adapted_from == [parent.metadata.name]
    assert task.metadata.is_beta
    assert task.hf_subsets == ["english"]
    assert all(
        Path(source.filename).name == parent.prediction_file_name
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
    assert (
        task.first_stage_predictions["qwen-text-image"].document_representation
        == "text-image"
    )
    assert task.metadata.descriptive_stats == parent.metadata.descriptive_stats
    assert mteb.get_task(parent.metadata.name).metadata.model_dump() == before
    assert not task.data_loaded


@pytest.mark.parametrize(
    ("source", "expected"),
    [("bm25-text", "d1"), ("bge-text", "d2"), ("qwen-text-image", "d2")],
)
def test_prepared_source_uses_parent_data(tmp_path, monkeypatch, source, expected):
    task = mteb.get_task("Vidore3HrReranking", hf_subsets=["english"])
    parent = mteb.get_task("Vidore3HrRetrieval.v2")
    mock = MockRetrievalTask()
    mock.load_data()
    shared = mock.dataset["default"]["test"]
    qrels = shared["relevant_docs"]

    def load_data(self):
        self.dataset = {"english": {"test": shared}}
        self.data_loaded = True

    monkeypatch.setattr(type(parent), "load_data", load_data)
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
    task.convert_to_reranking(first_stage=source, top_k=1)
    assert shared["top_ranked"] == {"q1": [expected], "q2": [expected]}
    assert shared["relevant_docs"] is qrels
    assert task.reranking_configuration.first_stage == source
    assert (
        task.reranking_configuration.predictions.repo_id
        == "mteb/Vidore3RetrievalPredictions"
    )


def test_requires_candidate_selection_before_evaluation():
    task = mteb.get_task("Vidore3HrReranking")
    with pytest.raises(ValueError, match="Select candidates"):
        task.evaluate(None, encode_kwargs={})
