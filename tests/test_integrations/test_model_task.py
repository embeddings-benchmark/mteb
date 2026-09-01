"""test mteb.MTEB's integration with mock tasks across modalities (text, audio, image, video) and model
types (encoder, sparse encoder, late-interaction encoder, cross-encoder).

Only text and audio have a dedicated Reranking-type mock task (one that supplies `top_ranked`
candidates, required by CrossEncoderProtocol models); image and video mock task grids don't have
one yet, so there's no `test_benchmark_image_cross_encoder`/`test_benchmark_video_cross_encoder`
below.
"""

import logging
import sys
from dataclasses import dataclass, field

import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks import (
    MOCK_MAEB_TASK_GRID,
    MOCK_MIEB_TASK_GRID,
    MOCK_MVEB_TASK_GRID,
    MOCK_TASK_TEST_GRID,
    MockAsymVideoAudioPairClassificationTask,
    MockAsymVideoAudioPairClassificationTaskV2,
    MockAudioReranking,
    MockRerankingTask,
    MockSymCustomVideoAudioPairClassificationTaskV2,
    MockVideoAudioPairClassificationTask,
)
from mteb.mocks.mock_tasks import MockAsymCustomTextImagePairClassificationTaskV2

logging.basicConfig(level=logging.INFO)


@dataclass
class ModelInfo:
    name: str
    expected_scores: dict[str, float]
    model: mteb.EncoderProtocol = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = mteb.get_model(self.name)


DENSE_MODEL = ModelInfo(
    name="mteb/baseline-random-encoder",
    expected_scores={
        "MockMultilingualBitextMiningTask": 1.0,
        "MockMultilingualParallelBitextMiningTask": 1.0,
        "MockMultilingualClassificationTask": 1.0,
        "MockMultilingualClusteringTask": 1.0,
        "MockMultilingualClusteringFastTask": 1.0,
        "MockMultilingualPairClassificationTask": 0.5,
        "MockMultilingualRerankingTask": 0.75,
        "MockMultilingualRetrievalTask": 0.81546,
        "MockMultilingualSTSTask": -1.0,
        "MockMultilingualMultilabelClassification": 1.0,
        "MockMultilingualSummarizationTask": 0.0,
        "MockMultilingualInstructionRetrieval": 0.63093,
        "MockMultilingualInstructionReranking": 0.63093,
        "MockBitextMiningTask": 1.0,
        "MockClassificationTask": 1.0,
        "MockRegressionTask": 1.0,
        "MockClusteringTask": 1.0,
        "LegacyMockClusteringFastTask": 1.0,
        "MockPairClassificationTask": 0.5,
        "MockRerankingTask": 0.75,
        "MockRetrievalTask": 0.81546,
        "MockSTSTask": -1.0,
        "MockMultilabelClassification": 1.0,
        "MockSummarizationTask": 0.0,
        "MockInstructionRetrieval": 0.63093,
        "MockInstructionReranking": 0.63093,
        "MockRetrievalDialogTask": 0.81546,
        "MockTextZeroShotClassification": 1.0,
        "MockAudioClusteringTask": 1.0,
        "MockAudioMultilabelClassification": 1.0,
        "MockAudioZeroshotClassification": 1.0,
        "MockAny2AnyRetrievalT2A": 1.0,
        "MockAny2AnyRetrievalA2T": 1.0,
        "MockAny2AnyRetrievalA2A": 1.0,
        "MockAudioReranking": 1.0,
        "MockAudioClassification": 1.0,
        "MockAudioClassificationCrossVal": 1.0,
        "AbsTaskAudioPairClassification": 1.0,
        "MockAny2AnyRetrievalI2T": 0.63093,
        "MockAny2AnyRetrievalT2I": 0.81546,
        "MockVisionCentricQA": 1.0,
        "MockImageClassification": 1.0,
        "MockImageClustering": 1.0,
        "MockImageTextPairClassification": 1.0,
        "MockVisualSTS": float("nan"),
        "MockZeroShotClassification": 1.0,
        "MockImageMultilabelClassification": 1.0,
        "MockMultilingualImageClassification": 1.0,
        "MockMultilingualImageTextPairClassification": 1.0,
        "MockMultilingualVisionCentricQA": 1.0,
        "MockImageClusteringFastTask": 1.0,
        "MockImageRegressionTask": 1.0,
        "MockPairImageClassificationTask": 1.0,
        "MockAsymCustomTextImagePairClassificationTaskV2": 1.0,
        "MockSymCustomVideoAudioPairClassificationTaskV2": 1.0,
        "MockAsymVideoAudioPairClassificationTaskV2": 1.0,
        "MockAsymVideoAudioPairClassificationTask": 1.0,
        "MockVideoAudioPairClassification": 1.0,
        "MockVideoClassification": 1.0,
        "MockVideoClusteringTask": 1.0,
        "MockVideoMultilabelClassification": 1.0,
        "MockVideoZeroshotClassification": 1.0,
        "MockVideoPairClassification": 1.0,
        "MockVideoRetrievalV2T": 0.81546,
        "MockVideoRetrievalT2V": 0.81546,
    },
)
SPARSE_MODEL = ModelInfo(
    name="mteb/baseline-random-sparse-encoder",
    expected_scores={
        "MockMultilingualBitextMiningTask": 0.5,
        "MockMultilingualParallelBitextMiningTask": 0.75,
        "MockMultilingualClassificationTask": 1.0,
        "MockMultilingualClusteringTask": 1.0,
        "MockMultilingualClusteringFastTask": 1.0,
        "MockMultilingualPairClassificationTask": 0.5,
        "MockMultilingualRerankingTask": 0.75,
        "MockMultilingualRetrievalTask": 0.81546,
        "MockMultilingualSTSTask": -1.0,
        "MockMultilingualMultilabelClassification": 1.0,
        "MockMultilingualSummarizationTask": 0.0,
        "MockMultilingualInstructionRetrieval": 0.63093,
        "MockMultilingualInstructionReranking": 0.63093,
        "MockBitextMiningTask": 0.5,
        "MockClassificationTask": 1.0,
        "MockRegressionTask": 1.0,
        "MockClusteringTask": 1.0,
        "LegacyMockClusteringFastTask": 1.0,
        "MockPairClassificationTask": 0.5,
        "MockRerankingTask": 0.75,
        "MockRetrievalTask": 0.81546,
        "MockSTSTask": -1.0,
        "MockMultilabelClassification": 1.0,
        "MockSummarizationTask": 0.0,
        "MockInstructionRetrieval": 0.63093,
        "MockInstructionReranking": 0.63093,
        "MockRetrievalDialogTask": 1.0,
        "MockTextZeroShotClassification": 1.0,
        "MockAudioClusteringTask": 1.0,
        "MockAudioMultilabelClassification": 1.0,
        "MockAudioZeroshotClassification": 0.0,
        "MockAny2AnyRetrievalT2A": 0.81546,
        "MockAny2AnyRetrievalA2T": 0.81546,
        "MockAny2AnyRetrievalA2A": 1.0,
        "MockAudioReranking": 1.0,
        "MockAudioClassification": 1.0,
        "MockAudioClassificationCrossVal": 1.0,
        "AbsTaskAudioPairClassification": 0.5,
        "MockAny2AnyRetrievalI2T": 0.81546,
        "MockAny2AnyRetrievalT2I": 1.0,
        "MockVisionCentricQA": 1.0,
        "MockImageClassification": 1.0,
        "MockImageClustering": 1.0,
        "MockImageTextPairClassification": 1.0,
        "MockVisualSTS": float("nan"),
        "MockZeroShotClassification": 0.5,
        "MockImageMultilabelClassification": 1.0,
        "MockMultilingualImageClassification": 1.0,
        "MockMultilingualImageTextPairClassification": 1.0,
        "MockMultilingualVisionCentricQA": 1.0,
        "MockImageClusteringFastTask": 1.0,
        "MockImageRegressionTask": -1.0,
        "MockPairImageClassificationTask": 1.0,
        "MockVideoClassification": 1.0,
        "MockVideoClusteringTask": 1.0,
        "MockVideoMultilabelClassification": 1.0,
        "MockVideoZeroshotClassification": 0.0,
        "MockVideoPairClassification": 0.5,
        "MockVideoRetrievalV2T": 1.0,
        "MockVideoRetrievalT2V": 1.0,
    },
)
COLBERT_MODEL = ModelInfo(
    name="mteb/baseline-random-colbert",
    expected_scores={
        "MockMultilingualRerankingTask": 0.75,
        "MockMultilingualRetrievalTask": 0.81546,
        "MockMultilingualInstructionRetrieval": 0.63093,
        "MockMultilingualInstructionReranking": 0.63093,
        "MockRerankingTask": 0.75,
        "MockRetrievalTask": 0.81546,
        "MockInstructionRetrieval": 0.63093,
        "MockInstructionReranking": 0.63093,
        "MockRetrievalDialogTask": 1.0,
        "MockAny2AnyRetrievalT2A": 0.81546,
        "MockAny2AnyRetrievalA2T": 0.81546,
        "MockAny2AnyRetrievalA2A": 1.0,
        "MockAudioReranking": 1.0,
        "MockAny2AnyRetrievalI2T": 0.81546,
        "MockAny2AnyRetrievalT2I": 0.63093,
        "MockVisionCentricQA": 1.0,
        "MockMultilingualVisionCentricQA": 1.0,
        "MockVideoRetrievalV2T": 0.81546,
        "MockVideoRetrievalT2V": 0.81546,
    },
)
CROSS_ENCODER_MODEL = ModelInfo(
    name="mteb/baseline-random-cross-encoder",
    expected_scores={
        "MockRerankingTask": 0.75,
        "MockAudioReranking": 1.0,
    },
)

# Audio/video decoding differs between the macOS and Linux/Windows codec stacks. Because the
# random baselines hash decoded media bytes, keep strict expected scores for each codec family.
if sys.platform == "darwin":
    DENSE_MODEL.expected_scores.update(
        {
            "MockAudioZeroshotClassification": 0.5,
            "AbsTaskAudioPairClassification": 0.5,
            "MockVideoZeroshotClassification": 0.5,
            "MockVideoPairClassification": 0.5,
            "MockVideoRetrievalV2T": 0.63093,
            "MockVideoRetrievalT2V": 0.63093,
            "MockSymCustomVideoAudioPairClassificationTaskV2": 0.5,
            "MockAsymVideoAudioPairClassificationTaskV2": 0.5,
            "MockAsymVideoAudioPairClassificationTask": 0.5,
        }
    )
    SPARSE_MODEL.expected_scores.update(
        {
            "MockAudioZeroshotClassification": 0.5,
            "MockVideoZeroshotClassification": 0.5,
            "MockVideoRetrievalV2T": 0.63093,
            "MockVideoRetrievalT2V": 0.63093,
        }
    )
    COLBERT_MODEL.expected_scores.update(
        {
            "MockAny2AnyRetrievalT2A": 0.63093,
            "MockVideoRetrievalV2T": 0.63093,
        }
    )


def _evaluate_and_assert_score(
    task: AbsTask,
    model_info: ModelInfo,
) -> None:
    # Parametrized task objects are created at collection time and reused by tests assigned
    # to the same xdist worker. Evaluate a fresh task so scores do not depend on test order.
    task = type(task)()
    results = mteb.evaluate(model_info.model, task, cache=None)
    result = results[0]
    expected_score = model_info.expected_scores[result.task_name]

    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model_info.name} final score changed for {result.task_name}"
    )


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_text_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_text_sparse_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_TASK_TEST_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
def test_benchmark_text_colbert(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(task, COLBERT_MODEL)


@pytest.mark.parametrize("task", [MockRerankingTask()], ids=lambda t: t.metadata.name)
def test_benchmark_text_cross_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(task, CROSS_ENCODER_MODEL)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_audio_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_audio_sparse_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MAEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
def test_benchmark_audio_colbert(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(task, COLBERT_MODEL)


@pytest.mark.parametrize("task", [MockAudioReranking()], ids=lambda t: t.metadata.name)
def test_benchmark_audio_cross_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(task, CROSS_ENCODER_MODEL)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_image_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    _evaluate_and_assert_score(task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_image_sparse_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    _evaluate_and_assert_score(task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MIEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
def test_benchmark_image_colbert(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    _evaluate_and_assert_score(task, COLBERT_MODEL)


@pytest.mark.parametrize("task", MOCK_MVEB_TASK_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_video_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_MVEB_TASK_GRID, ids=lambda t: t.metadata.name)
def test_benchmark_video_sparse_encoder(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MVEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
def test_benchmark_video_colbert(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(task, COLBERT_MODEL)


@pytest.mark.parametrize(
    "task",
    [
        MockAsymCustomTextImagePairClassificationTaskV2(),
        MockSymCustomVideoAudioPairClassificationTaskV2(),
        MockAsymVideoAudioPairClassificationTaskV2(),
        MockAsymVideoAudioPairClassificationTask(),
        MockVideoAudioPairClassificationTask(),
    ],
    ids=lambda t: t.metadata.name,
)
def test_benchmark_pair_classification(task: AbsTask):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(task, DENSE_MODEL)
