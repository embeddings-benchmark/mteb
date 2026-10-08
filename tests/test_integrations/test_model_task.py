"""test mteb.MTEB's integration with mock tasks across modalities (text, audio, image, video) and model
types (encoder, sparse encoder, late-interaction encoder, cross-encoder).

Each ``ModelInfo`` lists the expected final score for every task it can run; iterating it
yields the ``pytest.mark.parametrize`` cases. The ColBERT and cross-encoder baselines only
list the retrieval/reranking tasks they support, and ``assert_final_score`` skips a case
when the modality's optional dependency (``torchvision``/``torchaudio``) is missing.
"""

import logging
import sys

import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks import (
    LegacyMockClusteringFastTask,
    MockAny2AnyRetrievalA2ATask,
    MockAny2AnyRetrievalA2TTask,
    MockAny2AnyRetrievalI2TTask,
    MockAny2AnyRetrievalT2ATask,
    MockAny2AnyRetrievalT2ITask,
    MockAsymVideoAudioPairClassificationTask,
    MockAsymVideoAudioPairClassificationTaskV2,
    MockAudioClassification,
    MockAudioClassificationCrossVal,
    MockAudioClusteringTask,
    MockAudioMultilabelClassificationTask,
    MockAudioPairClassification,
    MockAudioReranking,
    MockAudioZeroshotClassificationTask,
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringTask,
    MockImageClassificationTask,
    MockImageClusteringFastTask,
    MockImageClusteringTask,
    MockImageMultilabelClassificationTask,
    MockImageRegressionTask,
    MockImageTextPairClassificationTask,
    MockInstructionReranking,
    MockInstructionRetrieval,
    MockMultiChoiceTask,
    MockMultilabelClassification,
    MockMultilingualBitextMiningTask,
    MockMultilingualClassificationTask,
    MockMultilingualClusteringFastTask,
    MockMultilingualClusteringTask,
    MockMultilingualImageClassificationTask,
    MockMultilingualImageTextPairClassificationTask,
    MockMultilingualInstructionReranking,
    MockMultilingualInstructionRetrieval,
    MockMultilingualMultiChoiceTask,
    MockMultilingualMultilabelClassification,
    MockMultilingualPairClassificationTask,
    MockMultilingualParallelBitextMiningTask,
    MockMultilingualRerankingTask,
    MockMultilingualRetrievalTask,
    MockMultilingualSTSTask,
    MockMultilingualSummarizationTask,
    MockPairClassificationTask,
    MockPairImageClassificationTask,
    MockRegressionTask,
    MockRerankingTask,
    MockRetrievalDialogTask,
    MockRetrievalTask,
    MockSTSTask,
    MockSummarizationTask,
    MockSymCustomVideoAudioPairClassificationTaskV2,
    MockTextZeroShotClassificationTask,
    MockVideoAudioPairClassificationTask,
    MockVideoClassification,
    MockVideoClusteringTask,
    MockVideoMultilabelClassificationTask,
    MockVideoPairClassificationTask,
    MockVideoRetrievalT2V,
    MockVideoRetrievalV2T,
    MockVideoZeroshotClassificationTask,
    MockVisualSTSTask,
    MockZeroShotClassificationTask,
)
from mteb.mocks.mock_tasks import MockAsymCustomTextImagePairClassificationTaskV2
from tests.test_integrations._model_info import ModelInfo, assert_final_score

logging.basicConfig(level=logging.INFO)

# ``expected_scores`` is keyed by mock task class. A few class names differ from the task
# name shown in the parametrize id, e.g. ``MockMultiChoiceTask`` -> "MockVisionCentricQA"
# and ``MockMultilingualMultiChoiceTask`` -> "MockMultilingualVisionCentricQA".

DENSE_MODEL = ModelInfo(
    name="mteb/baseline-random-encoder",
    expected_scores={
        MockMultilingualBitextMiningTask: 1.0,
        MockMultilingualParallelBitextMiningTask: 1.0,
        MockMultilingualClassificationTask: 1.0,
        MockMultilingualClusteringTask: 1.0,
        MockMultilingualClusteringFastTask: 1.0,
        MockMultilingualPairClassificationTask: 0.5,
        MockMultilingualRerankingTask: 0.75,
        MockMultilingualRetrievalTask: 0.81546,
        MockMultilingualSTSTask: -1.0,
        MockMultilingualMultilabelClassification: 1.0,
        MockMultilingualSummarizationTask: 0.0,
        MockMultilingualInstructionRetrieval: 0.63093,
        MockMultilingualInstructionReranking: 0.63093,
        MockBitextMiningTask: 1.0,
        MockClassificationTask: 1.0,
        MockRegressionTask: 1.0,
        MockClusteringTask: 1.0,
        LegacyMockClusteringFastTask: 1.0,
        MockPairClassificationTask: 0.5,
        MockRerankingTask: 0.75,
        MockRetrievalTask: 0.81546,
        MockSTSTask: -1.0,
        MockMultilabelClassification: 1.0,
        MockSummarizationTask: 0.0,
        MockInstructionRetrieval: 0.63093,
        MockInstructionReranking: 0.63093,
        MockRetrievalDialogTask: 0.81546,
        MockTextZeroShotClassificationTask: 1.0,
        MockAudioClusteringTask: 1.0,
        MockAudioMultilabelClassificationTask: 1.0,
        MockAudioZeroshotClassificationTask: 1.0,
        MockAny2AnyRetrievalT2ATask: 1.0,
        MockAny2AnyRetrievalA2TTask: 1.0,
        MockAny2AnyRetrievalA2ATask: 1.0,
        MockAudioReranking: 1.0,
        MockAudioClassification: 1.0,
        MockAudioClassificationCrossVal: 1.0,
        MockAudioPairClassification: 1.0,
        MockAny2AnyRetrievalI2TTask: 0.63093,
        MockAny2AnyRetrievalT2ITask: 0.81546,
        MockMultiChoiceTask: 1.0,
        MockImageClassificationTask: 1.0,
        MockImageClusteringTask: 1.0,
        MockImageTextPairClassificationTask: 1.0,
        MockVisualSTSTask: float("nan"),
        MockZeroShotClassificationTask: 1.0,
        MockImageMultilabelClassificationTask: 1.0,
        MockMultilingualImageClassificationTask: 1.0,
        MockMultilingualImageTextPairClassificationTask: 1.0,
        MockMultilingualMultiChoiceTask: 1.0,
        MockImageClusteringFastTask: 1.0,
        MockImageRegressionTask: 1.0,
        MockPairImageClassificationTask: 1.0,
        MockAsymCustomTextImagePairClassificationTaskV2: 1.0,
        MockSymCustomVideoAudioPairClassificationTaskV2: 1.0,
        MockAsymVideoAudioPairClassificationTaskV2: 1.0,
        MockAsymVideoAudioPairClassificationTask: 1.0,
        MockVideoAudioPairClassificationTask: 1.0,
        MockVideoClassification: 1.0,
        MockVideoClusteringTask: 1.0,
        MockVideoMultilabelClassificationTask: 1.0,
        MockVideoZeroshotClassificationTask: 1.0,
        MockVideoPairClassificationTask: 1.0,
        MockVideoRetrievalV2T: 0.81546,
        MockVideoRetrievalT2V: 0.81546,
    },
)
SPARSE_MODEL = ModelInfo(
    name="mteb/baseline-random-sparse-encoder",
    expected_scores={
        MockMultilingualBitextMiningTask: 0.5,
        MockMultilingualParallelBitextMiningTask: 0.75,
        MockMultilingualClassificationTask: 1.0,
        MockMultilingualClusteringTask: 1.0,
        MockMultilingualClusteringFastTask: 1.0,
        MockMultilingualPairClassificationTask: 0.5,
        MockMultilingualRerankingTask: 0.75,
        MockMultilingualRetrievalTask: 0.81546,
        MockMultilingualSTSTask: -1.0,
        MockMultilingualMultilabelClassification: 1.0,
        MockMultilingualSummarizationTask: 0.0,
        MockMultilingualInstructionRetrieval: 0.63093,
        MockMultilingualInstructionReranking: 0.63093,
        MockBitextMiningTask: 0.5,
        MockClassificationTask: 1.0,
        MockRegressionTask: 1.0,
        MockClusteringTask: 1.0,
        LegacyMockClusteringFastTask: 1.0,
        MockPairClassificationTask: 0.5,
        MockRerankingTask: 0.75,
        MockRetrievalTask: 0.81546,
        MockSTSTask: -1.0,
        MockMultilabelClassification: 1.0,
        MockSummarizationTask: 0.0,
        MockInstructionRetrieval: 0.63093,
        MockInstructionReranking: 0.63093,
        MockRetrievalDialogTask: 1.0,
        MockTextZeroShotClassificationTask: 1.0,
        MockAudioClusteringTask: 1.0,
        MockAudioMultilabelClassificationTask: 1.0,
        MockAudioZeroshotClassificationTask: 0.0,
        MockAny2AnyRetrievalT2ATask: 0.81546,
        MockAny2AnyRetrievalA2TTask: 0.81546,
        MockAny2AnyRetrievalA2ATask: 1.0,
        MockAudioReranking: 1.0,
        MockAudioClassification: 1.0,
        MockAudioClassificationCrossVal: 1.0,
        MockAudioPairClassification: 0.5,
        MockAny2AnyRetrievalI2TTask: 0.81546,
        MockAny2AnyRetrievalT2ITask: 1.0,
        MockMultiChoiceTask: 1.0,
        MockImageClassificationTask: 1.0,
        MockImageClusteringTask: 1.0,
        MockImageTextPairClassificationTask: 1.0,
        MockVisualSTSTask: float("nan"),
        MockZeroShotClassificationTask: 0.5,
        MockImageMultilabelClassificationTask: 1.0,
        MockMultilingualImageClassificationTask: 1.0,
        MockMultilingualImageTextPairClassificationTask: 1.0,
        MockMultilingualMultiChoiceTask: 1.0,
        MockImageClusteringFastTask: 1.0,
        MockImageRegressionTask: -1.0,
        MockPairImageClassificationTask: 1.0,
        MockVideoClassification: 1.0,
        MockVideoClusteringTask: 1.0,
        MockVideoMultilabelClassificationTask: 1.0,
        MockVideoZeroshotClassificationTask: 0.0,
        MockVideoPairClassificationTask: 0.5,
        MockVideoRetrievalV2T: 1.0,
        MockVideoRetrievalT2V: 1.0,
    },
)
COLBERT_MODEL = ModelInfo(
    name="mteb/baseline-random-colbert",
    expected_scores={
        MockMultilingualRerankingTask: 0.75,
        MockMultilingualRetrievalTask: 0.81546,
        MockMultilingualInstructionRetrieval: 0.63093,
        MockMultilingualInstructionReranking: 0.63093,
        MockRerankingTask: 0.75,
        MockRetrievalTask: 0.81546,
        MockInstructionRetrieval: 0.63093,
        MockInstructionReranking: 0.63093,
        MockRetrievalDialogTask: 1.0,
        MockAny2AnyRetrievalT2ATask: 0.81546,
        MockAny2AnyRetrievalA2TTask: 0.81546,
        MockAny2AnyRetrievalA2ATask: 1.0,
        MockAudioReranking: 1.0,
        MockAny2AnyRetrievalI2TTask: 0.81546,
        MockAny2AnyRetrievalT2ITask: 0.63093,
        MockMultiChoiceTask: 1.0,
        MockMultilingualMultiChoiceTask: 1.0,
        MockVideoRetrievalV2T: 0.81546,
        MockVideoRetrievalT2V: 0.81546,
    },
)
CROSS_ENCODER_MODEL = ModelInfo(
    name="mteb/baseline-random-cross-encoder",
    expected_scores={
        MockRerankingTask: 0.75,
        MockAudioReranking: 1.0,
    },
)

# Audio/video decoding differs between the macOS and Linux/Windows codec stacks. Because the
# random baselines hash decoded media bytes, keep strict expected scores for each codec family.
if sys.platform == "darwin":
    DENSE_MODEL.expected_scores.update(
        {
            MockAudioZeroshotClassificationTask: 0.5,
            MockAudioPairClassification: 0.5,
            MockVideoZeroshotClassificationTask: 0.5,
            MockVideoPairClassificationTask: 0.5,
            MockVideoRetrievalV2T: 0.63093,
            MockVideoRetrievalT2V: 0.63093,
            MockSymCustomVideoAudioPairClassificationTaskV2: 0.5,
            MockAsymVideoAudioPairClassificationTaskV2: 0.5,
            MockAsymVideoAudioPairClassificationTask: 0.5,
        }
    )
    SPARSE_MODEL.expected_scores.update(
        {
            MockAudioZeroshotClassificationTask: 0.5,
            MockVideoZeroshotClassificationTask: 0.5,
            MockVideoRetrievalV2T: 0.63093,
            MockVideoRetrievalT2V: 0.63093,
        }
    )
    COLBERT_MODEL.expected_scores.update(
        {
            MockAny2AnyRetrievalT2ATask: 0.63093,
            MockVideoRetrievalV2T: 0.63093,
        }
    )


@pytest.mark.parametrize(("model", "task", "expected_score"), DENSE_MODEL)
def test_benchmark_encoder(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), SPARSE_MODEL)
def test_benchmark_sparse_encoder(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), COLBERT_MODEL)
def test_benchmark_colbert(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), CROSS_ENCODER_MODEL)
def test_benchmark_cross_encoder(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)
