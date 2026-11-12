from pathlib import Path

import pytest
import sentence_transformers
from packaging.version import Version

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks.mock_tasks import MockRerankingTask, MockRetrievalTask
from mteb.models import MultiVectorWrapper
from mteb.models.sentence_transformer_wrapper import (
    SENTENCE_TRANSFORMERS_MULTI_VECTOR_VERSION,
)

# Constructed directly rather than via `mteb.get_model`: the mteb model registry may point
# `lightonai/LateOn` at a pylate-based loader instead (`pylate` isn't installed in every test
# environment), and this test is about `MultiVectorWrapper` itself, not about whichever
# loader happens to be registered for this particular model name. Revision pinned for
# reproducibility, independent of the registry's own pin.
_MODEL_REVISION = "6bb4488a7a1f1769f7a69fa1ff0c74c6a7b98cbd"


def _skip_if_multi_vector_unsupported() -> None:
    if (
        Version(sentence_transformers.__version__).release
        < Version(SENTENCE_TRANSFORMERS_MULTI_VECTOR_VERSION).release
    ):
        pytest.skip(
            f"sentence-transformers >= {SENTENCE_TRANSFORMERS_MULTI_VECTOR_VERSION} is required for MultiVectorEncoder"
        )


@pytest.mark.parametrize("model_name", ["lightonai/LateOn"])
@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_multi_vector_encoder_model_e2e(task: AbsTask, model_name: str, tmp_path: Path):
    """MultiVectorEncoder-backed model on a full-corpus retrieval task (brute-force search)."""
    _skip_if_multi_vector_unsupported()
    task._eval_splits = ["test"]

    model = MultiVectorWrapper(model_name, revision=_MODEL_REVISION)

    results = mteb.evaluate(model, task, cache=None)

    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.5


@pytest.mark.parametrize("model_name", ["lightonai/LateOn"])
@pytest.mark.parametrize("task", [MockRerankingTask()])
def test_multi_vector_encoder_model_reranking_e2e(
    task: AbsTask, model_name: str, tmp_path: Path
):
    """MultiVectorEncoder-backed model on a reranking task (pre-ranked `top_ranked` candidates)."""
    _skip_if_multi_vector_unsupported()
    task._eval_splits = ["test"]

    model = MultiVectorWrapper(model_name, revision=_MODEL_REVISION)
    results = mteb.evaluate(model, task, cache=None)

    result = results[0]
    assert result.scores["test"][0]["ndcg_at_1"] == 0.5
