import numpy as np
import pytest
from datasets import Dataset

import mteb
from mteb._create_dataloaders import create_dataloader
from mteb.cache import ResultCache
from mteb.evaluate import _check_model_modalities
from mteb.mocks import (
    MockVideoClassification,
    MockVideoClusteringTask,
    MockVideoRetrievalT2V,
    MockVideoRetrievalV2T,
    MockVideoZeroshotClassificationTask,
)
from mteb.mocks.mock_tasks.create_mock_samples import create_mock_video_bytes
from mteb.models import Video2ImagesWrapper
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_implementations.random_baseline import _image_to_vector
from mteb.models.video_wrappers import DEFAULT_NUM_FRAMES
from mteb.types import PromptType

pytest.importorskip("torchcodec", reason="Video dependencies are not installed")
pytest.importorskip("av", reason="Video dependencies are not installed")
pytest.importorskip(
    "datasets", minversion="4.0.0", reason="datasets.Video requires datasets>=4.0"
)

VIDEO_TASKS = [
    MockVideoRetrievalT2V(),
    MockVideoRetrievalV2T(),
    MockVideoZeroshotClassificationTask(),
    MockVideoClassification(),
    MockVideoClusteringTask(),
]
DEFAULT_WARNING = f"default {DEFAULT_NUM_FRAMES} frames per video"


def _model(modalities: list[str], model_type: list[str] | None = None):
    model = mteb.get_model("mteb/baseline-random-encoder")
    update: dict = {"modalities": modalities}
    if model_type is not None:
        update["model_type"] = model_type
    model.mteb_model_meta = model.mteb_model_meta.model_copy(update=update)
    return model


NON_DENSE_TYPES = ["late-interaction", "cross-encoder", "sparse"]


def _image_model():
    return _model(["text", "image"])


def test_requires_image_modality():
    with pytest.raises(ValueError, match="image"):
        Video2ImagesWrapper(_model(["text"]))


def test_rejects_non_positive_num_frames():
    with pytest.raises(ValueError, match="num_frames"):
        Video2ImagesWrapper(_image_model(), num_frames=0)


def test_rejects_num_frames_together_with_fps():
    with pytest.raises(ValueError, match="not both"):
        Video2ImagesWrapper(_image_model(), num_frames=4, fps=2.0)


def test_wrapper_meta_leaves_inner_model_untouched():
    model = _image_model()
    wrapper = Video2ImagesWrapper(model, num_frames=4)

    assert wrapper.mteb_model_meta.modalities == ["text", "image", "video"]
    assert wrapper.mteb_model_meta.experiment_kwargs["video_num_frames"] == 4
    assert wrapper.mteb_model_meta.experiment_kwargs["video_frame_pooling"] == "mean"
    assert model.mteb_model_meta.modalities == ["text", "image"]
    assert "video_num_frames" not in (model.mteb_model_meta.experiment_kwargs or {})


def test_default_num_frames_is_not_an_experiment():
    wrapper = Video2ImagesWrapper(_image_model())
    assert wrapper.num_frames == DEFAULT_NUM_FRAMES == 8
    assert wrapper.mteb_model_meta.modalities == ["text", "image", "video"]
    assert not wrapper.mteb_model_meta.experiment_kwargs
    assert wrapper.mteb_model_meta.experiment_name is None


def test_fps_mode_records_meta_without_num_frames():
    wrapper = Video2ImagesWrapper(_image_model(), fps=2.0, max_frames=16)
    experiment_kwargs = wrapper.mteb_model_meta.experiment_kwargs

    assert wrapper.num_frames is None
    assert experiment_kwargs["video_fps"] == 2.0
    assert experiment_kwargs["video_max_frames"] == 16
    assert "video_num_frames" not in experiment_kwargs


@pytest.mark.parametrize(
    "sampling", [{"num_frames": 4}, {"fps": 12.0}, {"fps": 12.0, "max_frames": 5}]
)
def test_pooled_embedding_is_mean_of_frame_embeddings(sampling):
    from datasets import Video
    from torchvision.transforms.functional import to_pil_image

    videos = Dataset.from_dict(
        {"video": create_mock_video_bytes(np.random.default_rng(0), n=3)}
    ).cast_column("video", Video())
    task = MockVideoRetrievalT2V()
    wrapper = Video2ImagesWrapper(_image_model(), **sampling)
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
        frames = FramesCollator.resample_video(row["video"], **sampling)
        frame_vectors = [
            _image_to_vector(to_pil_image(frame), embed_dim) for frame in frames
        ]
        expected.append(np.mean(frame_vectors, axis=0))
    np.testing.assert_allclose(
        np.asarray(embeddings), np.stack(expected), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("task", VIDEO_TASKS)
def test_evaluate_does_not_autowrap_image_models(task):
    with pytest.raises(ValueError, match="Video2ImagesWrapper"):
        mteb.evaluate(_image_model(), task, cache=None)


def test_evaluate_rejects_text_only_models_on_video_without_hint():
    with pytest.raises(ValueError, match="none overlap") as exc:
        mteb.evaluate(_model(["text"]), MockVideoRetrievalT2V(), cache=None)
    assert "Video2ImagesWrapper" not in str(exc.value)


@pytest.mark.parametrize("task", VIDEO_TASKS)
def test_evaluate_with_explicit_wrapper(task):
    wrapper = Video2ImagesWrapper(_image_model(), num_frames=4)
    mteb.evaluate(wrapper, task, cache=None)


@pytest.mark.parametrize("num_frames", [None, DEFAULT_NUM_FRAMES])
def test_default_protocol_is_stored_as_regular_results(tmp_path, num_frames):
    wrapper = Video2ImagesWrapper(_image_model(), num_frames=num_frames)
    mteb.evaluate(wrapper, MockVideoRetrievalT2V(), cache=ResultCache(tmp_path))
    (result_file,) = tmp_path.rglob("MockVideoRetrievalT2V.json")
    assert "experiments" not in result_file.parts


def test_other_frame_counts_are_stored_as_experiment(tmp_path):
    wrapper = Video2ImagesWrapper(_image_model(), num_frames=4)
    mteb.evaluate(wrapper, MockVideoRetrievalT2V(), cache=ResultCache(tmp_path))
    (result_file,) = tmp_path.rglob("MockVideoRetrievalT2V.json")
    assert result_file.parent.name == "video_frame_pooling_mean__video_num_frames_4"
    assert result_file.parent.parent.name == "experiments"


@pytest.mark.parametrize("model_type", NON_DENSE_TYPES)
def test_requires_dense_model(model_type):
    with pytest.raises(ValueError, match="dense encoder"):
        Video2ImagesWrapper(_model(["text", "image"], [model_type]))


def test_rejects_models_that_already_support_video():
    with pytest.raises(ValueError, match="already supports the 'video' modality"):
        Video2ImagesWrapper(_model(["text", "image", "video"]))


def test_partial_overlap_with_video_only_warns():
    """Tasks mixing video with modalities the model supports are not rejected."""
    task = mteb.get_task("XModBenchVT2TReranking")
    meta = mteb.get_model_meta("openai/clip-vit-base-patch32")
    _check_model_modalities(meta, task)


def test_video_task_error_points_to_wrapper():
    task = mteb.get_task("CoVRRVT2VRetrieval")
    meta = mteb.get_model_meta("openai/clip-vit-base-patch32")
    with pytest.raises(ValueError, match="Video2ImagesWrapper"):
        _check_model_modalities(meta, task)
