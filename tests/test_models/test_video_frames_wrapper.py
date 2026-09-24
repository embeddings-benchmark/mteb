import numpy as np
import pytest
from datasets import Dataset, Video
from torchvision.transforms.functional import to_pil_image

import mteb
from mteb._create_dataloaders import create_dataloader
from mteb.mocks import (
    MockVideoClassification,
    MockVideoClusteringTask,
    MockVideoRetrievalT2V,
    MockVideoRetrievalV2T,
    MockVideoZeroshotClassificationTask,
)
from mteb.mocks.mock_tasks.create_mock_samples import create_mock_video_bytes
from mteb.models import VideoFramesWrapper
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_implementations.random_baseline import _image_to_vector
from mteb.models.video_wrappers.video_frames_wrapper import DEFAULT_NUM_FRAMES
from mteb.types import PromptType

pytest.importorskip("torchcodec")
pytest.importorskip("av")


def _image_model():
    model = mteb.get_model("mteb/baseline-random-encoder")
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"modalities": ["text", "image"]}
    )
    return model


def test_requires_image_modality():
    model = mteb.get_model("mteb/baseline-random-encoder")
    model.mteb_model_meta = model.mteb_model_meta.model_copy(
        update={"modalities": ["text"]}
    )
    with pytest.raises(ValueError, match="image"):
        VideoFramesWrapper(model)


def test_rejects_non_positive_num_frames():
    with pytest.raises(ValueError, match="num_frames"):
        VideoFramesWrapper(_image_model(), num_frames=0)


def test_updates_model_meta():
    wrapper = VideoFramesWrapper(_image_model(), num_frames=4)
    meta = wrapper.mteb_model_meta
    assert meta.modalities == ["text", "image", "video"]
    assert meta.experiment_kwargs["video_num_frames"] == 4
    assert meta.experiment_kwargs["video_frame_pooling"] == "mean"


def test_default_num_frames():
    wrapper = VideoFramesWrapper(_image_model())
    assert wrapper.num_frames == DEFAULT_NUM_FRAMES == 8
    assert wrapper.mteb_model_meta.experiment_kwargs["video_num_frames"] == 8


def test_pooled_embedding_is_mean_of_frame_embeddings():
    num_frames = 4
    videos = Dataset.from_dict(
        {"video": create_mock_video_bytes(np.random.default_rng(0), n=3)}
    ).cast_column("video", Video())
    task = MockVideoRetrievalT2V()
    wrapper = VideoFramesWrapper(_image_model(), num_frames=num_frames)
    embed_dim = wrapper.model.embedding_dim

    loader = create_dataloader(
        videos,
        task_metadata=task.metadata,
        prompt_type=PromptType.document,
        batch_size=2,
    )
    embeddings = wrapper.encode(
        loader,
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.document,
    )
    assert embeddings.shape == (3, embed_dim)

    expected = []
    for row in videos:
        frames = FramesCollator.resample_video(row["video"], num_frames=num_frames)
        assert frames.shape[0] == num_frames
        frame_vectors = [
            _image_to_vector(to_pil_image(frame), embed_dim) for frame in frames
        ]
        expected.append(np.mean(frame_vectors, axis=0))
    np.testing.assert_allclose(
        np.asarray(embeddings), np.stack(expected), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    "task",
    [
        MockVideoRetrievalT2V(),
        MockVideoRetrievalV2T(),
        MockVideoZeroshotClassificationTask(),
        MockVideoClassification(),
        MockVideoClusteringTask(),
    ],
)
def test_evaluate_on_video_tasks(task):
    with pytest.raises(ValueError, match="none overlap"):
        mteb.evaluate(_image_model(), task, cache=None)

    wrapper = VideoFramesWrapper(_image_model(), num_frames=4)
    mteb.evaluate(wrapper, task, cache=None)
