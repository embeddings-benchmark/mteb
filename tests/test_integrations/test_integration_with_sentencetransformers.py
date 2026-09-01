"""test that mteb.evaluate integrates with SentenceTransformers"""

import logging
from dataclasses import dataclass, field

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


@dataclass
class ModelInfo:
    name: str
    expected_scores: dict[str, float]
    model_type: type[SentenceTransformer] | type[CrossEncoder] = field(repr=False)

    def load_model(self) -> SentenceTransformer | CrossEncoder:
        return self.model_type(self.name)


SENTENCE_TRANSFORMER_MODEL = ModelInfo(
    name="average_word_embeddings_levy_dependency",
    model_type=SentenceTransformer,
    expected_scores={
        "MockMultilingualBitextMiningTask": 0.5,
        "MockMultilingualParallelBitextMiningTask": 0.5,
        "MockMultilingualClassificationTask": 0.5,
        "MockMultilingualClusteringTask": 0.0,
        "MockMultilingualClusteringFastTask": 0.0,
        "MockMultilingualPairClassificationTask": 1.0,
        "MockMultilingualRerankingTask": 0.75,
        "MockMultilingualRetrievalTask": 0.81546,
        "MockMultilingualSTSTask": 1.0,
        "MockMultilingualMultilabelClassification": 1.0,
        "MockMultilingualSummarizationTask": float("nan"),
        "MockMultilingualInstructionRetrieval": 0.81546,
        "MockMultilingualInstructionReranking": 0.81546,
        "MockBitextMiningTask": 0.5,
        "MockClassificationTask": 0.5,
        "MockRegressionTask": float("nan"),
        "MockClusteringTask": 0.0,
        "LegacyMockClusteringFastTask": 0.0,
        "MockPairClassificationTask": 1.0,
        "MockRerankingTask": 0.75,
        "MockRetrievalTask": 0.81546,
        "MockSTSTask": 1.0,
        "MockMultilabelClassification": 1.0,
        "MockSummarizationTask": float("nan"),
        "MockInstructionRetrieval": 0.81546,
        "MockInstructionReranking": 0.81546,
        "MockRetrievalDialogTask": 0.81546,
        "MockTextZeroShotClassification": 0.5,
    },
)
CROSS_ENCODER_MODEL = ModelInfo(
    name="cross-encoder/ms-marco-TinyBERT-L2-v2",
    model_type=CrossEncoder,
    expected_scores={
        "MockRerankingTask": 0.5,
        "MockInstructionReranking": 0.63093,
    },
)


def _evaluate_and_assert_score(
    task: AbsTask,
    model_info: ModelInfo,
) -> None:
    model = model_info.load_model()
    if isinstance(model, SentenceTransformer):
        # Prior to https://github.com/embeddings-benchmark/mteb/pull/3079 the
        # SentenceTransformerWrapper would set the model's prompts to None because
        # the mock tasks are not in the MTEB task registry. The linked PR changes
        # this behavior and keeps the prompts as configured by the model, so this
        # test sets the prompts to an empty dict explicitly to preserve the legacy
        # behavior and focus the test on the tasks instead of the prompts.
        # Using empty dict instead of None to avoid TypeError in SentenceTransformers 5.0.0+
        model.prompts = {}

    task = type(task)()
    result = mteb.evaluate(model, task, cache=None)[0]
    expected_score = model_info.expected_scores[result.task_name]

    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model_info.name} final score changed for {result.task_name}"
    )


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
def test_sentence_transformer_integration(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(task, SENTENCE_TRANSFORMER_MODEL)


@pytest.mark.parametrize(
    "task",
    [
        MockRerankingTask(),
        MockInstructionReranking(),
    ],
    ids=lambda t: t.metadata.name,
)
def test_sentence_transformer_integration_cross_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(task, CROSS_ENCODER_MODEL)


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
