"""test mteb.MTEB's integration with mock tasks across modalities (text, audio, image, video) and model
types (encoder, sparse encoder, late-interaction encoder, cross-encoder).

Only text and audio have a dedicated Reranking-type mock task (one that supplies `top_ranked`
candidates, required by CrossEncoderProtocol models); image and video mock task grids don't have
one yet, so there's no `test_benchmark_image_cross_encoder`/`test_benchmark_video_cross_encoder`
below.
"""

import logging
import sys

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

DENSE_MODEL = "mteb/baseline-random-encoder"
SPARSE_MODEL = "mteb/baseline-random-sparse-encoder"
COLBERT_MODEL = "mteb/baseline-random-colbert"
CROSS_ENCODER_MODEL = "mteb/baseline-random-cross-encoder"

EXPECTED_SCORES = {
    (DENSE_MODEL, "MockMultilingualBitextMiningTask"): 1.0,
    (DENSE_MODEL, "MockMultilingualParallelBitextMiningTask"): 1.0,
    (DENSE_MODEL, "MockMultilingualClassificationTask"): 1.0,
    (DENSE_MODEL, "MockMultilingualClusteringTask"): 1.0,
    (DENSE_MODEL, "MockMultilingualClusteringFastTask"): 1.0,
    (DENSE_MODEL, "MockMultilingualPairClassificationTask"): 0.5,
    (DENSE_MODEL, "MockMultilingualRerankingTask"): 0.75,
    (DENSE_MODEL, "MockMultilingualRetrievalTask"): 0.81546,
    (DENSE_MODEL, "MockMultilingualSTSTask"): -1.0,
    (DENSE_MODEL, "MockMultilingualMultilabelClassification"): 1.0,
    (DENSE_MODEL, "MockMultilingualSummarizationTask"): 0.0,
    (DENSE_MODEL, "MockMultilingualInstructionRetrieval"): 0.63093,
    (DENSE_MODEL, "MockMultilingualInstructionReranking"): 0.63093,
    (DENSE_MODEL, "MockBitextMiningTask"): 1.0,
    (DENSE_MODEL, "MockClassificationTask"): 1.0,
    (DENSE_MODEL, "MockRegressionTask"): 1.0,
    (DENSE_MODEL, "MockClusteringTask"): 1.0,
    (DENSE_MODEL, "LegacyMockClusteringFastTask"): 1.0,
    (DENSE_MODEL, "MockPairClassificationTask"): 0.5,
    (DENSE_MODEL, "MockRerankingTask"): 0.75,
    (DENSE_MODEL, "MockRetrievalTask"): 0.81546,
    (DENSE_MODEL, "MockSTSTask"): -1.0,
    (DENSE_MODEL, "MockMultilabelClassification"): 1.0,
    (DENSE_MODEL, "MockSummarizationTask"): 0.0,
    (DENSE_MODEL, "MockInstructionRetrieval"): 0.63093,
    (DENSE_MODEL, "MockInstructionReranking"): 0.63093,
    (DENSE_MODEL, "MockRetrievalDialogTask"): 0.81546,
    (DENSE_MODEL, "MockTextZeroShotClassification"): 1.0,
    (DENSE_MODEL, "MockAudioClusteringTask"): 1.0,
    (DENSE_MODEL, "MockAudioMultilabelClassification"): 1.0,
    (DENSE_MODEL, "MockAudioZeroshotClassification"): 1.0,
    (DENSE_MODEL, "MockAny2AnyRetrievalT2A"): 1.0,
    (DENSE_MODEL, "MockAny2AnyRetrievalA2T"): 1.0,
    (DENSE_MODEL, "MockAny2AnyRetrievalA2A"): 1.0,
    (DENSE_MODEL, "MockAudioReranking"): 1.0,
    (DENSE_MODEL, "MockAudioClassification"): 1.0,
    (DENSE_MODEL, "MockAudioClassificationCrossVal"): 1.0,
    (DENSE_MODEL, "AbsTaskAudioPairClassification"): 1.0,
    (DENSE_MODEL, "MockAny2AnyRetrievalI2T"): 0.63093,
    (DENSE_MODEL, "MockAny2AnyRetrievalT2I"): 0.81546,
    (DENSE_MODEL, "MockVisionCentricQA"): 1.0,
    (DENSE_MODEL, "MockImageClassification"): 1.0,
    (DENSE_MODEL, "MockImageClustering"): 1.0,
    (DENSE_MODEL, "MockImageTextPairClassification"): 1.0,
    (DENSE_MODEL, "MockVisualSTS"): float("nan"),
    (DENSE_MODEL, "MockZeroShotClassification"): 1.0,
    (DENSE_MODEL, "MockImageMultilabelClassification"): 1.0,
    (DENSE_MODEL, "MockMultilingualImageClassification"): 1.0,
    (DENSE_MODEL, "MockMultilingualImageTextPairClassification"): 1.0,
    (DENSE_MODEL, "MockMultilingualVisionCentricQA"): 1.0,
    (DENSE_MODEL, "MockImageClusteringFastTask"): 1.0,
    (DENSE_MODEL, "MockImageRegressionTask"): 1.0,
    (DENSE_MODEL, "MockPairImageClassificationTask"): 1.0,
    (DENSE_MODEL, "MockAsymCustomTextImagePairClassificationTaskV2"): 1.0,
    (DENSE_MODEL, "MockSymCustomVideoAudioPairClassificationTaskV2"): 1.0,
    (DENSE_MODEL, "MockAsymVideoAudioPairClassificationTaskV2"): 1.0,
    (DENSE_MODEL, "MockAsymVideoAudioPairClassificationTask"): 1.0,
    (DENSE_MODEL, "MockVideoAudioPairClassification"): 1.0,
    (DENSE_MODEL, "MockVideoClassification"): 1.0,
    (DENSE_MODEL, "MockVideoClusteringTask"): 1.0,
    (DENSE_MODEL, "MockVideoMultilabelClassification"): 1.0,
    (DENSE_MODEL, "MockVideoZeroshotClassification"): 1.0,
    (DENSE_MODEL, "MockVideoPairClassification"): 1.0,
    (DENSE_MODEL, "MockVideoRetrievalV2T"): 0.81546,
    (DENSE_MODEL, "MockVideoRetrievalT2V"): 0.81546,
    (SPARSE_MODEL, "MockMultilingualBitextMiningTask"): 0.5,
    (SPARSE_MODEL, "MockMultilingualParallelBitextMiningTask"): 0.75,
    (SPARSE_MODEL, "MockMultilingualClassificationTask"): 1.0,
    (SPARSE_MODEL, "MockMultilingualClusteringTask"): 1.0,
    (SPARSE_MODEL, "MockMultilingualClusteringFastTask"): 1.0,
    (SPARSE_MODEL, "MockMultilingualPairClassificationTask"): 0.5,
    (SPARSE_MODEL, "MockMultilingualRerankingTask"): 0.75,
    (SPARSE_MODEL, "MockMultilingualRetrievalTask"): 0.81546,
    (SPARSE_MODEL, "MockMultilingualSTSTask"): -1.0,
    (SPARSE_MODEL, "MockMultilingualMultilabelClassification"): 1.0,
    (SPARSE_MODEL, "MockMultilingualSummarizationTask"): 0.0,
    (SPARSE_MODEL, "MockMultilingualInstructionRetrieval"): 0.63093,
    (SPARSE_MODEL, "MockMultilingualInstructionReranking"): 0.63093,
    (SPARSE_MODEL, "MockBitextMiningTask"): 0.5,
    (SPARSE_MODEL, "MockClassificationTask"): 1.0,
    (SPARSE_MODEL, "MockRegressionTask"): 1.0,
    (SPARSE_MODEL, "MockClusteringTask"): 1.0,
    (SPARSE_MODEL, "LegacyMockClusteringFastTask"): 1.0,
    (SPARSE_MODEL, "MockPairClassificationTask"): 0.5,
    (SPARSE_MODEL, "MockRerankingTask"): 0.75,
    (SPARSE_MODEL, "MockRetrievalTask"): 0.81546,
    (SPARSE_MODEL, "MockSTSTask"): -1.0,
    (SPARSE_MODEL, "MockMultilabelClassification"): 1.0,
    (SPARSE_MODEL, "MockSummarizationTask"): 0.0,
    (SPARSE_MODEL, "MockInstructionRetrieval"): 0.63093,
    (SPARSE_MODEL, "MockInstructionReranking"): 0.63093,
    (SPARSE_MODEL, "MockRetrievalDialogTask"): 1.0,
    (SPARSE_MODEL, "MockTextZeroShotClassification"): 1.0,
    (SPARSE_MODEL, "MockAudioClusteringTask"): 1.0,
    (SPARSE_MODEL, "MockAudioMultilabelClassification"): 1.0,
    (SPARSE_MODEL, "MockAudioZeroshotClassification"): 0.0,
    (SPARSE_MODEL, "MockAny2AnyRetrievalT2A"): 0.81546,
    (SPARSE_MODEL, "MockAny2AnyRetrievalA2T"): 0.81546,
    (SPARSE_MODEL, "MockAny2AnyRetrievalA2A"): 1.0,
    (SPARSE_MODEL, "MockAudioReranking"): 1.0,
    (SPARSE_MODEL, "MockAudioClassification"): 1.0,
    (SPARSE_MODEL, "MockAudioClassificationCrossVal"): 1.0,
    (SPARSE_MODEL, "AbsTaskAudioPairClassification"): 0.5,
    (SPARSE_MODEL, "MockAny2AnyRetrievalI2T"): 0.81546,
    (SPARSE_MODEL, "MockAny2AnyRetrievalT2I"): 1.0,
    (SPARSE_MODEL, "MockVisionCentricQA"): 1.0,
    (SPARSE_MODEL, "MockImageClassification"): 1.0,
    (SPARSE_MODEL, "MockImageClustering"): 1.0,
    (SPARSE_MODEL, "MockImageTextPairClassification"): 1.0,
    (SPARSE_MODEL, "MockVisualSTS"): float("nan"),
    (SPARSE_MODEL, "MockZeroShotClassification"): 0.5,
    (SPARSE_MODEL, "MockImageMultilabelClassification"): 1.0,
    (SPARSE_MODEL, "MockMultilingualImageClassification"): 1.0,
    (SPARSE_MODEL, "MockMultilingualImageTextPairClassification"): 1.0,
    (SPARSE_MODEL, "MockMultilingualVisionCentricQA"): 1.0,
    (SPARSE_MODEL, "MockImageClusteringFastTask"): 1.0,
    (SPARSE_MODEL, "MockImageRegressionTask"): -1.0,
    (SPARSE_MODEL, "MockPairImageClassificationTask"): 1.0,
    (SPARSE_MODEL, "MockVideoClassification"): 1.0,
    (SPARSE_MODEL, "MockVideoClusteringTask"): 1.0,
    (SPARSE_MODEL, "MockVideoMultilabelClassification"): 1.0,
    (SPARSE_MODEL, "MockVideoZeroshotClassification"): 0.0,
    (SPARSE_MODEL, "MockVideoPairClassification"): 0.5,
    (SPARSE_MODEL, "MockVideoRetrievalV2T"): 1.0,
    (SPARSE_MODEL, "MockVideoRetrievalT2V"): 1.0,
    (COLBERT_MODEL, "MockMultilingualRerankingTask"): 0.75,
    (COLBERT_MODEL, "MockMultilingualRetrievalTask"): 0.81546,
    (COLBERT_MODEL, "MockMultilingualInstructionRetrieval"): 0.63093,
    (COLBERT_MODEL, "MockMultilingualInstructionReranking"): 0.63093,
    (COLBERT_MODEL, "MockRerankingTask"): 0.75,
    (COLBERT_MODEL, "MockRetrievalTask"): 0.81546,
    (COLBERT_MODEL, "MockInstructionRetrieval"): 0.63093,
    (COLBERT_MODEL, "MockInstructionReranking"): 0.63093,
    (COLBERT_MODEL, "MockRetrievalDialogTask"): 1.0,
    (COLBERT_MODEL, "MockAny2AnyRetrievalT2A"): 0.81546,
    (COLBERT_MODEL, "MockAny2AnyRetrievalA2T"): 0.81546,
    (COLBERT_MODEL, "MockAny2AnyRetrievalA2A"): 1.0,
    (COLBERT_MODEL, "MockAudioReranking"): 1.0,
    (COLBERT_MODEL, "MockAny2AnyRetrievalI2T"): 0.81546,
    (COLBERT_MODEL, "MockAny2AnyRetrievalT2I"): 0.63093,
    (COLBERT_MODEL, "MockVisionCentricQA"): 1.0,
    (COLBERT_MODEL, "MockMultilingualVisionCentricQA"): 1.0,
    (COLBERT_MODEL, "MockVideoRetrievalV2T"): 0.81546,
    (COLBERT_MODEL, "MockVideoRetrievalT2V"): 0.81546,
    (CROSS_ENCODER_MODEL, "MockRerankingTask"): 0.75,
    (CROSS_ENCODER_MODEL, "MockAudioReranking"): 1.0,
}

# Audio/video decoding differs between the macOS and Linux/Windows codec stacks. Because the
# random baselines hash decoded media bytes, keep strict expected scores for each codec family.
if sys.platform == "darwin":
    EXPECTED_SCORES.update(
        {
            (DENSE_MODEL, "MockAudioZeroshotClassification"): 0.5,
            (DENSE_MODEL, "AbsTaskAudioPairClassification"): 0.5,
            (SPARSE_MODEL, "MockAudioZeroshotClassification"): 0.5,
            (COLBERT_MODEL, "MockAny2AnyRetrievalT2A"): 0.63093,
            (DENSE_MODEL, "MockVideoZeroshotClassification"): 0.5,
            (DENSE_MODEL, "MockVideoPairClassification"): 0.5,
            (DENSE_MODEL, "MockVideoRetrievalV2T"): 0.63093,
            (DENSE_MODEL, "MockVideoRetrievalT2V"): 0.63093,
            (DENSE_MODEL, "MockSymCustomVideoAudioPairClassificationTaskV2"): 0.5,
            (DENSE_MODEL, "MockAsymVideoAudioPairClassificationTaskV2"): 0.5,
            (DENSE_MODEL, "MockAsymVideoAudioPairClassificationTask"): 0.5,
            (SPARSE_MODEL, "MockVideoZeroshotClassification"): 0.5,
            (SPARSE_MODEL, "MockVideoRetrievalV2T"): 0.63093,
            (SPARSE_MODEL, "MockVideoRetrievalT2V"): 0.63093,
            (COLBERT_MODEL, "MockVideoRetrievalV2T"): 0.63093,
        }
    )


def _evaluate_and_assert_score(
    model: mteb.EncoderProtocol,
    task: AbsTask,
    model_name: str,
) -> None:
    # Parametrized task objects are created at collection time and reused by tests assigned
    # to the same xdist worker. Evaluate a fresh task so scores do not depend on test order.
    task = type(task)()
    results = mteb.evaluate(model, task, cache=None)
    result = results[0]
    expected_score = EXPECTED_SCORES[(model_name, result.task_name)]

    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model_name} final score changed for {result.task_name}"
    )


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL)])
def test_benchmark_text_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(SPARSE_MODEL)])
def test_benchmark_text_sparse_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_TASK_TEST_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(COLBERT_MODEL)])
def test_benchmark_text_colbert(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, COLBERT_MODEL)


@pytest.mark.parametrize("task", [MockRerankingTask()], ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(CROSS_ENCODER_MODEL)])
def test_benchmark_text_cross_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, CROSS_ENCODER_MODEL)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL)])
def test_benchmark_audio_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(model, task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(SPARSE_MODEL)])
def test_benchmark_audio_sparse_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(model, task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MAEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(COLBERT_MODEL)])
def test_benchmark_audio_colbert(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(model, task, COLBERT_MODEL)


@pytest.mark.parametrize("task", [MockAudioReranking()], ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(CROSS_ENCODER_MODEL)])
def test_benchmark_audio_cross_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    _evaluate_and_assert_score(model, task, CROSS_ENCODER_MODEL)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL)])
def test_benchmark_image_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    _evaluate_and_assert_score(model, task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(SPARSE_MODEL)])
def test_benchmark_image_sparse_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    _evaluate_and_assert_score(model, task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MIEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(COLBERT_MODEL)])
def test_benchmark_image_colbert(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    _evaluate_and_assert_score(model, task, COLBERT_MODEL)


@pytest.mark.parametrize("task", MOCK_MVEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL)])
def test_benchmark_video_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(model, task, DENSE_MODEL)


@pytest.mark.parametrize("task", MOCK_MVEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(SPARSE_MODEL)])
def test_benchmark_video_sparse_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(model, task, SPARSE_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MVEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(COLBERT_MODEL)])
def test_benchmark_video_colbert(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(model, task, COLBERT_MODEL)


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
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL)])
def test_benchmark_pair_classification(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    _evaluate_and_assert_score(model, task, DENSE_MODEL)
