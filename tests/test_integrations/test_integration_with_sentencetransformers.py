"""test that mteb.evaluate integrates with SentenceTransformers"""

import logging

import pytest
import sentence_transformers
from packaging.version import Version
from sentence_transformers import CrossEncoder, SentenceTransformer

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks import MOCK_TASK_TEST_GRID
from mteb.mocks.mock_tasks import (
    MockInstructionReranking,
    MockRerankingTask,
)
from mteb.models import ModelMeta

logging.basicConfig(level=logging.INFO)

SENTENCE_TRANSFORMER_MODEL = "average_word_embeddings_levy_dependency"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

EXPECTED_SCORES = {
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualBitextMiningTask"): 0.5,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualParallelBitextMiningTask"): 0.5,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualClassificationTask"): 0.5,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualClusteringTask"): 0.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualClusteringFastTask"): 0.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualPairClassificationTask"): 1.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualRerankingTask"): 0.75,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualRetrievalTask"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualSTSTask"): 1.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualMultilabelClassification"): 1.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualSummarizationTask"): float("nan"),
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualInstructionRetrieval"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilingualInstructionReranking"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockBitextMiningTask"): 0.5,
    (SENTENCE_TRANSFORMER_MODEL, "MockClassificationTask"): 0.5,
    (SENTENCE_TRANSFORMER_MODEL, "MockRegressionTask"): float("nan"),
    (SENTENCE_TRANSFORMER_MODEL, "MockClusteringTask"): 0.0,
    (SENTENCE_TRANSFORMER_MODEL, "LegacyMockClusteringFastTask"): 0.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockPairClassificationTask"): 1.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockRerankingTask"): 0.75,
    (SENTENCE_TRANSFORMER_MODEL, "MockRetrievalTask"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockSTSTask"): 1.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockMultilabelClassification"): 1.0,
    (SENTENCE_TRANSFORMER_MODEL, "MockSummarizationTask"): float("nan"),
    (SENTENCE_TRANSFORMER_MODEL, "MockInstructionRetrieval"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockInstructionReranking"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockRetrievalDialogTask"): 0.81546,
    (SENTENCE_TRANSFORMER_MODEL, "MockTextZeroShotClassification"): 0.5,
    (CROSS_ENCODER_MODEL, "MockRerankingTask"): 0.5,
    (CROSS_ENCODER_MODEL, "MockInstructionReranking"): 0.63093,
}


def _evaluate_and_assert_score(
    model: SentenceTransformer | CrossEncoder,
    task: AbsTask,
    model_name: str,
) -> None:
    task = type(task)()
    result = mteb.evaluate(model, task, cache=None)[0]
    expected_score = EXPECTED_SCORES[(model_name, result.task_name)]

    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model_name} final score changed for {result.task_name}"
    )


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model_name", [SENTENCE_TRANSFORMER_MODEL])
def test_sentence_transformer_integration(task: AbsTask, model_name: str):
    """Test that a task can be fetched and produces the expected final score."""
    model = SentenceTransformer(model_name)
    # Prior to https://github.com/embeddings-benchmark/mteb/pull/3079 the
    # SentenceTransformerWrapper would set the model's prompts to None because
    # the mock tasks are not in the MTEB task registry. The linked PR changes
    # this behavior and keeps the prompts as configured by the model, so this
    # test now sets the prompts to an empty dict explicitly to preserve the legacy
    # behavior and focus the test on the tasks instead of the prompts.
    # Using empty dict instead of None to avoid TypeError in SentenceTransformers 5.0.0+
    model.prompts = {}
    _evaluate_and_assert_score(model, task, model_name)


@pytest.mark.parametrize(
    "task",
    [
        MockRerankingTask(),
        MockInstructionReranking(),
    ],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model_name", [CROSS_ENCODER_MODEL])
def test_sentence_transformer_integration_cross_encoder(task: AbsTask, model_name: str):
    """Test that a task can be fetched and produces the expected final score."""
    model = CrossEncoder(model_name)
    _evaluate_and_assert_score(model, task, model_name)


def test_model_meta_load_sentence_transformer_metadata_from_model():
    # used also in test CLI
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    meta = ModelMeta.from_sentence_transformer_model(model)

    assert meta.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta.revision is not None
    assert meta.max_tokens == 256
    assert meta.embed_dim == 384
    if Version(sentence_transformers.__version__) >= Version("4.0.0"):
        assert meta.similarity_fn_name is not None
        assert meta.similarity_fn_name.value == "cosine"


@pytest.mark.parametrize("model_name", ["sentence-transformers/all-MiniLM-L6-v2"])
def test_model_meta_sentence_transformer_from_hub(model_name: str):
    meta = ModelMeta.from_hub(model_name)

    assert meta.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta.revision is not None
    assert meta.release_date == "2021-08-30"
    assert meta.n_parameters == 22713728
    assert meta.memory_usage_mb == 87
    assert meta.embed_dim == 384
    assert meta.license == "apache-2.0"
    assert meta.similarity_fn_name is not None
    assert meta.similarity_fn_name.value == "cosine"
    assert meta.max_tokens == 512


@pytest.mark.parametrize("model_name", ["cross-encoder/ms-marco-TinyBERT-L2-v2"])
def test_cross_encoder_model_meta(model_name: str):
    model = CrossEncoder(model_name)
    meta = ModelMeta.from_cross_encoder(model)

    assert meta.name == "cross-encoder/ms-marco-TinyBERT-L2-v2"
    assert meta.revision is not None
