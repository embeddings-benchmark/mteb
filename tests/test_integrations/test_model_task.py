"""test mteb.MTEB's integration with mock tasks across modalities (text, audio, image, video) and model
types (encoder, sparse encoder, late-interaction encoder, cross-encoder).

Only text and audio have a dedicated Reranking-type mock task (one that supplies `top_ranked`
candidates, required by CrossEncoderProtocol models); image and video mock task grids don't have
one yet, so there's no `test_benchmark_image_cross_encoder`/`test_benchmark_video_cross_encoder`
below.
"""

import logging

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
from mteb.mocks.mock_tasks import MockSymCustomTextImagePairClassificationTaskV2

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-encoder")])
def test_benchmark_text_encoder(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize(
    "model", [mteb.get_model("mteb/baseline-random-sparse-encoder")]
)
def test_benchmark_text_sparse_encoder(
    task: str | AbsTask, model: mteb.EncoderProtocol
):
    """Test that a task can be fetched and run"""
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_TASK_TEST_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-colbert")])
def test_benchmark_text_colbert(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", [MockRerankingTask()], ids=lambda t: t.metadata.name)
@pytest.mark.parametrize(
    "model", [mteb.get_model("mteb/baseline-random-cross-encoder")]
)
def test_benchmark_text_cross_encoder(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-encoder")])
def test_benchmark_audio_encoder(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MAEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize(
    "model", [mteb.get_model("mteb/baseline-random-sparse-encoder")]
)
def test_benchmark_audio_sparse_encoder(
    task: str | AbsTask, model: mteb.EncoderProtocol
):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MAEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-colbert")])
def test_benchmark_audio_colbert(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", [MockAudioReranking()], ids=lambda t: t.metadata.name)
@pytest.mark.parametrize(
    "model", [mteb.get_model("mteb/baseline-random-cross-encoder")]
)
def test_benchmark_audio_cross_encoder(
    task: str | AbsTask, model: mteb.EncoderProtocol
):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-encoder")])
def test_benchmark_image_encoder(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MIEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize(
    "model", [mteb.get_model("mteb/baseline-random-sparse-encoder")]
)
def test_benchmark_image_sparse_encoder(
    task: str | AbsTask, model: mteb.EncoderProtocol
):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MIEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-colbert")])
def test_benchmark_image_colbert(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Image dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MVEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-encoder")])
def test_benchmark_video_encoder(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize("task", MOCK_MVEB_TASK_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize(
    "model", [mteb.get_model("mteb/baseline-random-sparse-encoder")]
)
def test_benchmark_video_sparse_encoder(
    task: str | AbsTask, model: mteb.EncoderProtocol
):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize(
    "task",
    [t for t in MOCK_MVEB_TASK_GRID if t.metadata.simplified_task_type == "retrieval"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-colbert")])
def test_benchmark_video_colbert(task: str | AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    mteb.evaluate(model, task, cache=None)


@pytest.mark.parametrize(
    "task",
    [
        MockSymCustomTextImagePairClassificationTaskV2(),
        MockSymCustomVideoAudioPairClassificationTaskV2(),
        MockAsymVideoAudioPairClassificationTaskV2(),
        MockAsymVideoAudioPairClassificationTask(),
        MockVideoAudioPairClassificationTask(),
    ],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model("mteb/baseline-random-encoder")])
def test_benchmark_pair_classification(
    task: str | AbsTask, model: mteb.EncoderProtocol
):
    """Test that a task can be fetched and run"""
    pytest.importorskip("torchvision", reason="Video dependencies are not installed")
    pytest.importorskip("torchaudio", reason="Video dependencies are not installed")
    mteb.evaluate(model, task, cache=None)
