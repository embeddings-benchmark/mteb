import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.mocks.mock_tasks import (
    MockRerankingTask,
    MockRetrievalTask,
)
from mteb.models import SearchEncoderWrapper
from mteb.models.model_meta import ScoringFunction
from mteb.models.search_encoder_index import FaissSearchIndex


@pytest.mark.parametrize(
    "task",
    [
        MockRetrievalTask(),
        MockRerankingTask(),
    ],
)
@pytest.mark.parametrize(
    "similarity",
    [ScoringFunction.DOT_PRODUCT, ScoringFunction.COSINE, ScoringFunction.EUCLIDEAN],
)
def test_retrieval_backends(
    task: AbsTaskRetrieval, similarity: ScoringFunction, tmp_path: Path
):
    """Test different retrieval backends for retrieval and reranking tasks."""
    pytest.importorskip("faiss", reason="faiss is not installed")
    model = mteb.get_model("mteb/baseline-random-encoder")
    model_meta = deepcopy(model.mteb_model_meta)
    model_meta = model_meta.model_copy(update={"similarity_fn_name": similarity})
    model.mteb_model_meta = model_meta

    faiss_backend = SearchEncoderWrapper(model, index_backend=FaissSearchIndex(model))

    python_backend_predictions = tmp_path / "python_backend_predictions"
    faiss_backend_predictions = tmp_path / "faiss_backend_predictions"

    python_results = mteb.evaluate(
        model,
        task,
        prediction_folder=python_backend_predictions,
        cache=None,
    )
    faiss_results = mteb.evaluate(
        faiss_backend,
        task,
        prediction_folder=faiss_backend_predictions,
        cache=None,
    )

    assert (
        python_results.task_results[0].get_score()
        == faiss_results.task_results[0].get_score()
    )

    with task._predictions_path(python_backend_predictions).open(encoding="utf-8") as f:
        full_python_predictions = json.load(f)
        python_predictions = full_python_predictions["default"]["test"]

    with task._predictions_path(faiss_backend_predictions).open(encoding="utf-8") as f:
        full_faiss_predictions = json.load(f)
        faiss_predictions = full_faiss_predictions["default"]["test"]

    for python_pred_key, faiss_pred_key in zip(
        sorted(python_predictions.keys()), sorted(faiss_predictions.keys()), strict=True
    ):
        assert python_pred_key == faiss_pred_key
        assert python_predictions[python_pred_key] == pytest.approx(
            faiss_predictions[faiss_pred_key]
        )


@pytest.mark.parametrize(
    "similarity",
    [ScoringFunction.DOT_PRODUCT, ScoringFunction.COSINE, ScoringFunction.EUCLIDEAN],
)
@pytest.mark.parametrize("top_k", [1, 5])
@pytest.mark.parametrize(
    "candidates",
    [
        {"q1": ["d3", "d1"], "q2": ["d2", "d3"]},
        {"q1": ["d3", "d1", "d2"], "q2": ["d2"]},
        {"q1": ["d3", "d1", "d2"], "q2": []},
    ],
    ids=["equal", "unequal", "empty"],
)
def test_faiss_reranking_candidate_lengths(
    similarity: ScoringFunction, top_k: int, candidates: dict[str, list[str]]
) -> None:
    """Keep each query's scores and candidate positions, including empty rows."""
    pytest.importorskip("faiss", reason="faiss is not installed")
    model = mteb.get_model("mteb/baseline-random-encoder")
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"similarity_fn_name": similarity}
    )
    documents = np.array([[1, 0], [3, 1], [4, 4]], dtype=np.float32)
    queries = np.array([[1, 0], [2, 1]], dtype=np.float32)
    document_ids = ["d1", "d2", "d3"]
    backend = FaissSearchIndex(model)
    backend.add_documents(documents, document_ids)

    warning = (
        pytest.warns(UserWarning, match="No top-ranked documents for query q2")
        if not candidates["q2"]
        else nullcontext()
    )
    with warning:
        scores, indices = backend.search(
            queries,
            top_k,
            similarity_fn=model.similarity,
            top_ranked=candidates,
            query_idx_to_id={0: "q1", 1: "q2"},
        )

    for query_index, query_id in enumerate(["q1", "q2"]):
        reference = []
        for candidate_index, document_id in enumerate(candidates[query_id]):
            query = queries[query_index].astype(np.float64)
            document = documents[document_ids.index(document_id)].astype(np.float64)
            if similarity == ScoringFunction.EUCLIDEAN:
                score = -np.linalg.norm(query - document)
            elif similarity == ScoringFunction.COSINE:
                score = np.dot(query, document) / (
                    np.linalg.norm(query) * np.linalg.norm(document)
                )
            else:
                score = np.dot(query, document)
            reference.append((float(score), candidate_index))
        reference.sort(reverse=True)
        reference = reference[:top_k]
        assert indices[query_index] == [index for _, index in reference]
        assert scores[query_index] == pytest.approx([score for score, _ in reference])


def test_faiss_euclidean_reranking_unequal_candidates(tmp_path: Path) -> None:
    """Public evaluation agrees with direct scoring for unequal candidate lists."""
    pytest.importorskip("faiss", reason="faiss is not installed")
    task = MockRerankingTask()
    task.load_data()
    task.dataset["default"]["test"]["top_ranked"]["q1"] = ["d1"]
    model = mteb.get_model("mteb/baseline-random-encoder")
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"similarity_fn_name": ScoringFunction.EUCLIDEAN}
    )
    predictions = []
    for name, encoder in [
        ("direct", model),
        ("faiss", SearchEncoderWrapper(model, index_backend=FaissSearchIndex(model))),
    ]:
        prediction_folder = tmp_path / name
        mteb.evaluate(encoder, task, prediction_folder=prediction_folder, cache=None)
        predictions.append(
            json.loads(
                task._predictions_path(prediction_folder).read_text(encoding="utf-8")
            )["default"]["test"]
        )
    assert set(predictions[0]["q1"]) == {"d1"}
    for query_id in predictions[0]:
        assert predictions[1][query_id] == pytest.approx(predictions[0][query_id])
