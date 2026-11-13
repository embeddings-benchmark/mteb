import json
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
@pytest.mark.parametrize("num_queries", [0, 2])
def test_faiss_search_empty_corpus(
    similarity: ScoringFunction, num_queries: int
) -> None:
    """An indexed empty corpus has no matches for any query."""
    pytest.importorskip("faiss", reason="faiss is not installed")
    model = mteb.get_model("mteb/baseline-random-encoder")
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"similarity_fn_name": similarity}
    )
    backend = FaissSearchIndex(model)
    backend.add_documents(np.empty((0, 2), dtype=np.float32), [])

    scores, indices = backend.search(
        np.ones((num_queries, 2), dtype=np.float32),
        top_k=5,
        similarity_fn=model.similarity,
    )

    assert scores == [[] for _ in range(num_queries)]
    assert indices == [[] for _ in range(num_queries)]


@pytest.mark.parametrize("second_candidates", [["d2"], []], ids=["unequal", "empty"])
def test_faiss_euclidean_reranking_candidate_lengths(
    second_candidates: list[str],
) -> None:
    """Euclidean scores support unequal candidate counts, including empty rows."""
    pytest.importorskip("faiss", reason="faiss is not installed")
    model = mteb.get_model("mteb/baseline-random-encoder")
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"similarity_fn_name": ScoringFunction.EUCLIDEAN}
    )
    backend = FaissSearchIndex(model)
    backend.add_documents(
        np.array([[1, 0], [3, 1], [4, 4]], dtype=np.float32),
        ["d1", "d2", "d3"],
    )

    scores, indices = backend.search(
        np.array([[1, 0], [2, 1]], dtype=np.float32),
        top_k=5,
        similarity_fn=model.similarity,
        top_ranked={"q1": ["d3", "d1", "d2"], "q2": second_candidates},
        query_idx_to_id={0: "q1", 1: "q2"},
    )

    # Indices refer to positions within each query's candidate list.
    assert indices == [[1, 2, 0], [0] if second_candidates else []]
    assert scores[0] == pytest.approx([0, -np.sqrt(5), -5])
    assert scores[1] == pytest.approx([-1] if second_candidates else [])


def test_faiss_reranking_missing_query_in_top_ranked(tmp_path: Path) -> None:
    """Evaluation preserves empty results for queries without reranking candidates."""
    pytest.importorskip("faiss", reason="faiss is not installed")
    task = MockRerankingTask()
    task.load_data()
    del task.dataset["default"]["test"]["top_ranked"]["q2"]
    model = mteb.get_model("mteb/baseline-random-encoder")
    wrapper = SearchEncoderWrapper(model, index_backend=FaissSearchIndex(model))

    with pytest.warns(UserWarning, match="No top-ranked documents for query q2"):
        mteb.evaluate(wrapper, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open(encoding="utf-8") as f:
        predictions = json.load(f)["default"]["test"]

    assert set(predictions) == {"q1", "q2"}
    assert set(predictions["q1"]) == {"d1", "d2"}
    assert predictions["q2"] == {}
