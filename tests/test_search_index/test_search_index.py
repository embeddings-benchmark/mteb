import json
from copy import deepcopy
from pathlib import Path

import pytest

import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.models import SearchEncoderWrapper
from mteb.models.model_meta import ScoringFunction
from mteb.models.search_encoder_index import FaissSearchIndex
from tests.mock_tasks import (
    MockRerankingTask,
    MockRetrievalTask,
)


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
    model = mteb.get_model("baseline/random-encoder-baseline")
    model_meta = deepcopy(model.mteb_model_meta)
    model_meta.similarity_fn_name = similarity
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

    with task._predictions_path(python_backend_predictions).open() as f:
        full_python_predictions = json.load(f)
        python_predictions = full_python_predictions["default"]["test"]

    with task._predictions_path(faiss_backend_predictions).open() as f:
        full_faiss_predictions = json.load(f)
        faiss_predictions = full_faiss_predictions["default"]["test"]

    for python_pred_key, faiss_pred_key in zip(
        sorted(python_predictions.keys()), sorted(faiss_predictions.keys())
    ):
        assert python_pred_key == faiss_pred_key
        assert python_predictions[python_pred_key] == pytest.approx(
            faiss_predictions[faiss_pred_key]
        )
