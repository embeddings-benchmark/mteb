from __future__ import annotations

import io
import sys
import wave
from types import ModuleType

import numpy as np
import pytest
import torch

from mteb.models.modality_collators import FramesCollator
from mteb.models.model_implementations import google_gemini
from mteb.models.model_implementations.google_gemini import (
    GoogleGeminiEmbeddingModel,
    _audio_to_wav_bytes,
    _build_gemini_content,
    _format_gemini_embedding_2_text,
    _video_to_mp4_bytes,
)
from mteb.types import PromptType


class _FakePart:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str) -> dict[str, bytes | str]:
        return {"data": data, "mime_type": mime_type}


def test_gemini_audio_accepts_exactly_180_seconds() -> None:
    audio = {
        "array": np.zeros(180, dtype=np.float32),
        "sampling_rate": 1,
    }

    assert _audio_to_wav_bytes(audio)


def test_gemini_audio_truncates_more_than_180_seconds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio = {
        "array": np.zeros(181, dtype=np.float32),
        "sampling_rate": 1,
    }

    wav_bytes = _audio_to_wav_bytes(audio)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 1
        assert wav_file.getnframes() == 180
    assert "Truncating Gemini Embedding 2 audio input" in caplog.text


@pytest.fixture
def fake_google_genai_types(monkeypatch: pytest.MonkeyPatch) -> None:
    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")
    types_module.Part = _FakePart  # type: ignore[attr-defined]
    google_module.genai = genai_module  # type: ignore[attr-defined]
    genai_module.types = types_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


@pytest.mark.parametrize(
    ("google_task_type", "expected"),
    [
        ("RETRIEVAL_QUERY", "task: search result | query: example text"),
        ("QUESTION_ANSWERING", "task: question answering | query: example text"),
        ("FACT_VERIFICATION", "task: fact checking | query: example text"),
        ("CLASSIFICATION", "task: classification | query: example text"),
        ("CLUSTERING", "task: clustering | query: example text"),
        ("SEMANTIC_SIMILARITY", "task: sentence similarity | query: example text"),
    ],
)
def test_gemini_embedding_2_formats_task_prefixes(
    google_task_type: str, expected: str
) -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", google_task_type, PromptType.query
        )
        == expected
    )


def test_gemini_embedding_2_formats_documents_with_title() -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", "FACT_VERIFICATION", PromptType.document, "Example"
        )
        == "title: Example | text: example text"
    )


def test_gemini_embedding_2_formats_documents_without_title() -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", "FACT_VERIFICATION", PromptType.document
        )
        == "title: none | text: example text"
    )


def test_gemini_embedding_2_leaves_unknown_task_type_unchanged() -> None:
    assert _format_gemini_embedding_2_text("example text", None, None) == "example text"


def test_build_gemini_content_formats_text_only(fake_google_genai_types: None) -> None:
    content = _build_gemini_content(
        text="example text",
        title="Example",
        image=None,
        audio=None,
        video=None,
        google_task_type="RETRIEVAL_DOCUMENT",
        prompt_type=PromptType.document,
    )

    assert content == "title: Example | text: example text"


def test_build_gemini_content_preserves_cross_modal_text(
    fake_google_genai_types: None,
) -> None:
    content = _build_gemini_content(
        text="example text",
        title=None,
        image=None,
        audio=None,
        video=None,
        google_task_type="RETRIEVAL_QUERY",
        prompt_type=PromptType.query,
        use_text_formatting=False,
    )

    assert content == "example text"


def test_build_gemini_content_aggregates_modalities(
    fake_google_genai_types: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = object()
    audio = {"array": np.zeros(16, dtype=np.float32), "sampling_rate": 16_000}
    video = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
    monkeypatch.setattr(
        google_gemini, "_video_to_mp4_bytes", lambda *_args, **_kwargs: b"mp4"
    )

    content = _build_gemini_content(
        text="example text",
        title=None,
        image=image,
        audio=audio,
        video=video,
        google_task_type="RETRIEVAL_QUERY",
        prompt_type=PromptType.query,
    )

    assert isinstance(content, list)
    assert content[0] == "example text"
    assert content[1] is image
    assert content[2]["mime_type"] == "audio/wav"
    assert content[2]["data"].startswith(b"RIFF")
    assert content[3] == {"data": b"mp4", "mime_type": "video/mp4"}


@pytest.mark.parametrize("media", ["image", "audio", "video"])
def test_build_gemini_content_preserves_raw_text_with_media(
    media: str,
    fake_google_genai_types: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {"image": None, "audio": None, "video": None}
    if media == "image":
        kwargs["image"] = object()
    elif media == "audio":
        kwargs["audio"] = {
            "array": np.zeros(16, dtype=np.float32),
            "sampling_rate": 16_000,
        }
    else:
        kwargs["video"] = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
        monkeypatch.setattr(
            google_gemini, "_video_to_mp4_bytes", lambda *_args, **_kwargs: b"mp4"
        )

    content = _build_gemini_content(
        text="example text",
        title="Example title",
        google_task_type="RETRIEVAL_DOCUMENT",
        prompt_type=PromptType.document,
        **kwargs,  # type: ignore[arg-type]
    )

    assert isinstance(content, list)
    assert content[0] == "example text"


def test_build_gemini_content_rejects_empty_row(fake_google_genai_types: None) -> None:
    with pytest.raises(ValueError, match="No supported Gemini input modality"):
        _build_gemini_content(
            text=None,
            title=None,
            image=None,
            audio=None,
            video=None,
            google_task_type=None,
            prompt_type=None,
        )


def test_encode_preserves_existing_audio_collator(
    fake_google_genai_types: None,
) -> None:
    original_collate_fn = object()
    audio = {"array": np.zeros(16, dtype=np.float32), "sampling_rate": 8_000}

    class _AudioInputs:
        dataset = type("_Dataset", (), {"features": {"audio": object()}})()
        collate_fn = original_collate_fn

        def __iter__(self):
            return iter([{"audio": [audio]}])

    model = GoogleGeminiEmbeddingModel.__new__(GoogleGeminiEmbeddingModel)
    model.model_prompts = {}
    model.get_prompt_name = lambda *_args: "AudioTask"  # type: ignore[method-assign]
    model._embed = lambda *_args, **_kwargs: np.zeros((1, 3))  # type: ignore[method-assign]
    inputs = _AudioInputs()

    task_metadata = type("_TaskMetadata", (), {"modalities": ["audio"]})()
    model.encode(
        inputs,  # type: ignore[arg-type]
        task_metadata=task_metadata,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
    )

    assert inputs.collate_fn is original_collate_fn


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        (["text"], True),
        (["text", "image"], False),
    ],
)
def test_encode_formats_text_for_text_only_tasks(
    monkeypatch: pytest.MonkeyPatch,
    modalities: list[str],
    expected: bool,
) -> None:
    captured: dict = {}

    class _TextInputs:
        dataset = type("_Dataset", (), {"features": {"text": object()}})()

        def __iter__(self):
            return iter([{"text": ["example text"]}])

    def _build_content(**kwargs):
        captured.update(kwargs)
        return kwargs["text"]

    monkeypatch.setattr(google_gemini, "_build_gemini_content", _build_content)

    model = GoogleGeminiEmbeddingModel.__new__(GoogleGeminiEmbeddingModel)
    model.model_prompts = {"query": "RETRIEVAL_QUERY"}
    model.get_prompt_name = lambda *_args: "query"  # type: ignore[method-assign]
    model._embed = lambda *_args, **_kwargs: np.zeros((1, 3))  # type: ignore[method-assign]
    task_metadata = type("_TaskMetadata", (), {"modalities": modalities})()

    model.encode(
        _TextInputs(),  # type: ignore[arg-type]
        task_metadata=task_metadata,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )

    assert captured["use_text_formatting"] is expected


def test_encode_configures_video_collator_and_preserves_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class _VideoInputs:
        dataset = type(
            "_Dataset", (), {"features": {"text": object(), "video": object()}}
        )()
        collate_fn = None

        def __iter__(self):
            video = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)
            return iter(
                [
                    {"text": ["first", "second"], "video": [video, video]},
                    {"text": ["third"], "video": [video]},
                ]
            )

    monkeypatch.setattr(
        google_gemini,
        "_build_gemini_content",
        lambda **kwargs: kwargs["text"],
    )

    model = GoogleGeminiEmbeddingModel.__new__(GoogleGeminiEmbeddingModel)
    model.model_prompts = {}
    model.get_prompt_name = lambda *_args: "VideoTask"  # type: ignore[method-assign]

    def _embed(contents, *, show_progress_bar, batch_size):
        captured.update(
            contents=contents,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )
        return np.zeros((len(contents), 3))

    model._embed = _embed  # type: ignore[method-assign]
    inputs = _VideoInputs()

    task_metadata = type("_TaskMetadata", (), {"modalities": ["text", "video"]})()
    model.encode(
        inputs,  # type: ignore[arg-type]
        task_metadata=task_metadata,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
        batch_size=17,
    )

    assert isinstance(inputs.collate_fn, FramesCollator)
    assert inputs.collate_fn.fps == 1.0
    assert inputs.collate_fn.max_frames == 32
    assert captured == {
        "contents": ["first", "second", "third"],
        "show_progress_bar": False,
        "batch_size": 17,
    }


def test_video_to_mp4_bytes_encodes_normalized_frames() -> None:
    pytest.importorskip("torchcodec")
    from torchcodec.decoders import VideoDecoder

    frames = torch.ones((2, 3, 16, 18), dtype=torch.float32)

    video_bytes = _video_to_mp4_bytes(frames, fps=1.0)

    decoded = VideoDecoder(video_bytes).get_all_frames().data
    assert decoded.shape == (2, 3, 16, 18)
    assert decoded.float().mean() > 200


def test_video_to_mp4_bytes_pads_odd_dimensions() -> None:
    pytest.importorskip("torchcodec")
    from torchcodec.decoders import VideoDecoder

    frames = torch.zeros((2, 3, 15, 17), dtype=torch.uint8)

    video_bytes = _video_to_mp4_bytes(frames, fps=1.0)

    decoded = VideoDecoder(video_bytes).get_all_frames().data
    assert decoded.shape == (2, 3, 16, 18)


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (torch.zeros((3, 4, 4)), "shape"),
        (torch.zeros((0, 3, 4, 4)), "at least one"),
        (torch.zeros((1, 1, 4, 4)), "3 channels"),
        (torch.zeros((1, 3, 0, 4)), "too small"),
        (torch.zeros((1, 3, 4, 0)), "too small"),
    ],
)
def test_video_to_mp4_bytes_validates_frames(frames: torch.Tensor, match: str) -> None:
    pytest.importorskip("torchcodec")
    with pytest.raises(ValueError, match=match):
        _video_to_mp4_bytes(frames, fps=1.0)


def test_video_to_mp4_bytes_validates_fps() -> None:
    pytest.importorskip("torchcodec")
    with pytest.raises(ValueError, match="positive video FPS"):
        _video_to_mp4_bytes(torch.zeros((1, 3, 4, 4)), fps=0)
