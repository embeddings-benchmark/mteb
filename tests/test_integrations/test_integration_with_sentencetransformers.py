"""test that mteb.evaluate integrates with SentenceTransformers"""

import logging

import pytest
import sentence_transformers
from packaging.version import Version
from sentence_transformers import CrossEncoder, SentenceTransformer

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks import (
    LegacyMockClusteringFastTask,
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringTask,
    MockInstructionReranking,
    MockInstructionRetrieval,
    MockMultilabelClassification,
    MockMultilingualBitextMiningTask,
    MockMultilingualClassificationTask,
    MockMultilingualClusteringFastTask,
    MockMultilingualClusteringTask,
    MockMultilingualInstructionReranking,
    MockMultilingualInstructionRetrieval,
    MockMultilingualMultilabelClassification,
    MockMultilingualPairClassificationTask,
    MockMultilingualParallelBitextMiningTask,
    MockMultilingualRerankingTask,
    MockMultilingualRetrievalTask,
    MockMultilingualSTSTask,
    MockMultilingualSummarizationTask,
    MockPairClassificationTask,
    MockRegressionTask,
    MockRerankingTask,
    MockRetrievalDialogTask,
    MockRetrievalTask,
    MockSTSTask,
    MockSummarizationTask,
    MockTextZeroShotClassificationTask,
)
from mteb.models import ModelMeta
from tests.test_integrations._model_info import ModelInfo, assert_final_score

logging.basicConfig(level=logging.INFO)


def _load_sentence_transformer(name: str) -> SentenceTransformer:
    model = SentenceTransformer(name)
    # Prior to https://github.com/embeddings-benchmark/mteb/pull/3079 the
    # SentenceTransformerWrapper would set the model's prompts to None because
    # the mock tasks are not in the MTEB task registry. The linked PR changes
    # this behavior and keeps the prompts as configured by the model, so this
    # test clears the prompts explicitly to preserve the legacy behavior and
    # focus the test on the tasks instead of the prompts. Using an empty dict
    # instead of None avoids a TypeError in SentenceTransformers 5.0.0+.
    model.prompts = {}
    return model


SENTENCE_TRANSFORMER_MODEL = ModelInfo(
    name="average_word_embeddings_levy_dependency",
    loader=_load_sentence_transformer,
    expected_scores={
        MockMultilingualBitextMiningTask: 0.5,
        MockMultilingualParallelBitextMiningTask: 0.5,
        MockMultilingualClassificationTask: 0.5,
        MockMultilingualClusteringTask: 0.0,
        MockMultilingualClusteringFastTask: 0.0,
        MockMultilingualPairClassificationTask: 1.0,
        MockMultilingualRerankingTask: 0.75,
        MockMultilingualRetrievalTask: 0.81546,
        MockMultilingualSTSTask: 1.0,
        MockMultilingualMultilabelClassification: 1.0,
        MockMultilingualSummarizationTask: float("nan"),
        MockMultilingualInstructionRetrieval: 0.81546,
        MockMultilingualInstructionReranking: 0.81546,
        MockBitextMiningTask: 0.5,
        MockClassificationTask: 0.5,
        MockRegressionTask: float("nan"),
        MockClusteringTask: 0.0,
        LegacyMockClusteringFastTask: 0.0,
        MockPairClassificationTask: 1.0,
        MockRerankingTask: 0.75,
        MockRetrievalTask: 0.81546,
        MockSTSTask: 1.0,
        MockMultilabelClassification: 1.0,
        MockSummarizationTask: float("nan"),
        MockInstructionRetrieval: 0.81546,
        MockInstructionReranking: 0.81546,
        MockRetrievalDialogTask: 0.81546,
        MockTextZeroShotClassificationTask: 0.5,
    },
)
CROSS_ENCODER_MODEL = ModelInfo(
    name="cross-encoder/ms-marco-TinyBERT-L2-v2",
    loader=CrossEncoder,
    expected_scores={
        MockRerankingTask: 0.5,
        MockInstructionReranking: 0.63093,
    },
)


@pytest.mark.parametrize(
    ("model", "task", "expected_score"), SENTENCE_TRANSFORMER_MODEL
)
def test_sentence_transformer_integration(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), CROSS_ENCODER_MODEL)
def test_sentence_transformer_integration_cross_encoder(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


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
