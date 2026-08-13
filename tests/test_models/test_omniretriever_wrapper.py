from __future__ import annotations

from importlib.metadata import requires

import numpy as np
import pytest
import torch
from packaging.requirements import Requirement
from packaging.version import Version

import mteb
from mteb.models.model_implementations.omniretriever_models import (
    WAVE_BASE_MODEL,
    WAVE_BASE_REVISION,
    OmniRetrieverWrapper,
    omniretriever_7b,
)

AUDIO_PLACEHOLDER = "<|AUDIO|>"
VIDEO_PLACEHOLDER = "<|VIDEO|>"


class StubFeatureExtractor:
    """Stands in for the Whisper feature extractor."""

    def __call__(self, waveforms, **kwargs):
        n = len(waveforms)
        # 3000 mel frames is what "max_length" padding yields for Whisper.
        return {
            "input_features": torch.zeros(n, 128, 3000),
            "attention_mask": torch.ones(n, 3000, dtype=torch.long),
        }


class StubImageProcessor:
    temporal_patch_size = 2
    max_pixels = None
    min_pixels = None
    size: dict[str, int] = {}

    def __call__(self, images=None, videos=None, **kwargs):
        n = len(videos)
        return {
            "pixel_values_videos": torch.zeros(n, 4, 1176),
            "video_grid_thw": torch.tensor([[4, 2, 2]] * n),
        }


class StubProcessor:
    """Mimics the parts of Qwen2_5OmniProcessor the wrapper touches."""

    def __init__(self):
        self.feature_extractor = StubFeatureExtractor()
        self.image_processor = StubImageProcessor()
        self.tokenizer = StubTokenizer()
        self.replace_calls: list[dict] = []

    @staticmethod
    def apply_chat_template(conversation, add_generation_prompt=False, **kwargs):
        parts = []
        for item in conversation[0]["content"]:
            if item["type"] == "video":
                parts.append(f"<|vision_bos|>{VIDEO_PLACEHOLDER}<|vision_eos|>")
            elif item["type"] == "audio":
                parts.append(f"<|audio_bos|>{AUDIO_PLACEHOLDER}<|audio_eos|>")
            elif item["type"] == "text":
                parts.append(item["text"])
        body = "".join(parts)
        return [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n"
        ]

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        *,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        self.replace_calls.append(
            {
                "use_audio_in_video": use_audio_in_video,
                "position_id_per_seconds": position_id_per_seconds,
                "seconds_per_chunk": seconds_per_chunk,
                "video_second_per_grid": list(video_second_per_grid),
            }
        )
        out = []
        for row in text:
            expanded = row
            if AUDIO_PLACEHOLDER in expanded:
                expanded = expanded.replace(
                    AUDIO_PLACEHOLDER, AUDIO_PLACEHOLDER * next(audio_lengths, 3)
                )
            if VIDEO_PLACEHOLDER in expanded:
                next(video_grid_thw, None)
                expanded = expanded.replace(VIDEO_PLACEHOLDER, VIDEO_PLACEHOLDER * 4)
            out.append(expanded)
        return out


class StubTokenizer:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(
        self, prompts, padding=True, padding_side="right", return_tensors=None
    ):
        self.calls.append({"prompts": list(prompts), "padding_side": padding_side})
        length = max(len(p) for p in prompts)
        n = len(prompts)
        return {
            "input_ids": torch.ones(n, length, dtype=torch.long),
            "attention_mask": torch.ones(n, length, dtype=torch.long),
        }


class StubModel:
    """Returns unnormalised embeddings so the wrapper's normalisation is observable."""

    def __init__(self, dim=OmniRetrieverWrapper.EMBED_DIM):
        self.dim = dim
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch = kwargs["input_ids"].shape[0]
        generator = torch.Generator().manual_seed(0)
        embeds = (
            torch.randn(batch, self.dim, generator=generator) * 7.5 + 3.0
        )  # far from unit norm
        return type("Out", (), {"mllm_embeds": embeds})()


def make_wrapper() -> OmniRetrieverWrapper:
    """Build a wrapper without touching CUDA or downloading 7B weights."""
    wrapper = object.__new__(OmniRetrieverWrapper)
    wrapper.device = "cpu"
    wrapper.processor = StubProcessor()
    wrapper.tokenizer = wrapper.processor.tokenizer
    wrapper.model = StubModel()
    wrapper.max_audio_samples = (
        OmniRetrieverWrapper.MAX_AUDIO_SEC * OmniRetrieverWrapper.AUDIO_SAMPLING_RATE
    )
    return wrapper


def audio_item(seconds=8.0, sr=OmniRetrieverWrapper.AUDIO_SAMPLING_RATE):
    return {
        "array": np.zeros(int(seconds * sr), dtype=np.float32),
        "sampling_rate": sr,
    }


def video_frames(n=OmniRetrieverWrapper.NUM_FRAMES):
    return torch.randint(0, 255, (n, 3, 32, 32), dtype=torch.uint8)


# --------------------------------------------------------------------------- #
# Registration / metadata                                                     #
# --------------------------------------------------------------------------- #


def test_model_meta_is_registered():
    meta = mteb.get_model_meta("YunzeLiu/OmniRetriever-7B")
    assert meta is omniretriever_7b
    assert meta.loader is OmniRetrieverWrapper
    assert meta.embed_dim == OmniRetrieverWrapper.EMBED_DIM
    assert meta.similarity_fn_name.value == "cosine"


def test_meta_declares_supported_modalities():
    assert set(omniretriever_7b.modalities) == {"text", "audio", "video"}


def test_base_model_and_adapter_revisions_are_pinned():
    """Both Hub components must be pinned: the adapter is useless without the backbone."""
    assert omniretriever_7b.revision == "99328f1c5ce88695fa7070aac5b4a817aab60698"
    assert omniretriever_7b.adapted_from == WAVE_BASE_MODEL
    assert len(WAVE_BASE_REVISION) == 40
    assert len(omniretriever_7b.revision) == 40


def test_declared_extras_groups_are_valid_mteb_extras():
    """Groups are checked against the installed distribution's Provides-Extra."""
    groups = omniretriever_7b._resolve_extras_groups()
    omniretriever_7b._validate_extras_groups(groups)
    assert "omniretriever" in groups
    # torchcodec is not implied by any modality, unlike "audio"
    assert "video" in groups
    # appended automatically from modalities, so it need not be declared twice
    assert "audio" in groups
    assert "audio" not in omniretriever_7b.extra_requirements_groups


def _omniretriever_extra_requirements() -> list[Requirement]:
    """Requirements the 'omniretriever' extra contributes, from package metadata."""
    return [
        Requirement(r)
        for r in (requires("mteb") or [])
        if Requirement(r).marker is not None
        and Requirement(r).marker.evaluate({"extra": "omniretriever"})
    ]


def test_omniretriever_extra_pins_a_transformers_version_the_remote_code_supports():
    """WAVE-7B's remote code targets transformers 4.51.3 and fails to import on 5.x."""
    specs = {r.name: r.specifier for r in _omniretriever_extra_requirements()}
    assert "transformers" in specs, "the extra must pin transformers"
    specifier = specs["transformers"]
    assert Version("4.51.3") in specifier
    assert Version("4.56.0") in specifier
    assert Version("5.15.0") not in specifier
    assert Version("4.50.0") not in specifier


def test_omniretriever_extra_provides_peft():
    specs = {r.name: r.specifier for r in _omniretriever_extra_requirements()}
    assert "peft" in specs, "the adapter cannot be loaded without peft"


def test_n_parameters_matches_the_measured_loaded_model():
    """Backbone (9,410,651,007) + LoRA (6,881,280), measured on the real model."""
    assert omniretriever_7b.n_parameters == 9_410_651_007 + 6_881_280


# --------------------------------------------------------------------------- #
# Preprocessing contracts                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seconds", [4.0, 8.0, 22.0, 30.0])
def test_waveform_keeps_its_native_length(seconds):
    """Regression: capping audio at the training-time 8s truncates every Clotho
    clip (15-30s) to its first 8s and cost 3.6 R@1 on ClothoT2ARetrieval."""
    wrapper = make_wrapper()
    sr = OmniRetrieverWrapper.AUDIO_SAMPLING_RATE
    out = wrapper._prepare_waveform(audio_item(seconds))
    assert out.shape == (int(seconds * sr),)
    assert out.dtype == np.float32


def test_waveform_is_not_truncated_to_eight_seconds():
    wrapper = make_wrapper()
    sr = OmniRetrieverWrapper.AUDIO_SAMPLING_RATE
    ramp = np.arange(22 * sr, dtype=np.float32)
    out = wrapper._prepare_waveform({"array": ramp, "sampling_rate": sr})
    assert np.array_equal(out, ramp)


def test_pathologically_long_waveform_is_head_truncated_at_the_chunk_bound():
    """Only the 300s processor/backbone chunk boundary bounds length."""
    wrapper = make_wrapper()
    sr = OmniRetrieverWrapper.AUDIO_SAMPLING_RATE
    ramp = np.arange((OmniRetrieverWrapper.MAX_AUDIO_SEC + 10) * sr, dtype=np.float32)
    out = wrapper._prepare_waveform({"array": ramp, "sampling_rate": sr})
    assert np.array_equal(out, ramp[: wrapper.max_audio_samples])


def test_sub_second_waveform_is_zero_padded_to_one_second():
    """data_qwen.py: "pad audio to at least 1s"."""
    wrapper = make_wrapper()
    sr = OmniRetrieverWrapper.AUDIO_SAMPLING_RATE
    signal = np.ones(sr // 4, dtype=np.float32)
    out = wrapper._prepare_waveform({"array": signal, "sampling_rate": sr})
    assert out.shape == (sr,)
    assert np.array_equal(out[: sr // 4], signal)
    assert np.all(out[sr // 4 :] == 0.0)


def test_frames_converted_to_channels_last_uint8():
    frames = video_frames()
    out = OmniRetrieverWrapper._frames_to_thwc(frames)
    assert out.shape == (OmniRetrieverWrapper.NUM_FRAMES, 32, 32, 3)
    assert np.array_equal(out, frames.permute(0, 2, 3, 1).numpy())


# --------------------------------------------------------------------------- #
# Modality dispatch                                                           #
# --------------------------------------------------------------------------- #


def test_text_only_prompt_has_no_media_placeholders():
    wrapper = make_wrapper()
    prompt = wrapper._build_prompt("a dog barking", has_video=False, has_audio=False)
    assert "a dog barking" in prompt
    assert AUDIO_PLACEHOLDER not in prompt and VIDEO_PLACEHOLDER not in prompt
    # the chat template is stripped down to the bare user turn
    assert not prompt.startswith("<|im_start|>")
    assert prompt.endswith("<|im_end|>")


def test_video_only_prompt_uses_default_caption():
    wrapper = make_wrapper()
    prompt = wrapper._build_prompt(None, has_video=True, has_audio=False)
    assert VIDEO_PLACEHOLDER in prompt
    assert "Please describe the video." in prompt


def test_audio_only_prompt_uses_audio_tag():
    wrapper = make_wrapper()
    prompt = wrapper._build_prompt(None, has_video=False, has_audio=True)
    assert AUDIO_PLACEHOLDER in prompt
    assert VIDEO_PLACEHOLDER not in prompt


def test_audio_video_prompt_carries_only_the_video_tag():
    """Official pipeline folds audio into the video stream via use_audio_in_video."""
    wrapper = make_wrapper()
    prompt = wrapper._build_prompt(None, has_video=True, has_audio=True)
    assert VIDEO_PLACEHOLDER in prompt
    assert AUDIO_PLACEHOLDER not in prompt


def test_text_plus_media_prompts_keep_the_caption():
    wrapper = make_wrapper()
    tv = wrapper._build_prompt("a caption", has_video=True, has_audio=False)
    ta = wrapper._build_prompt("a caption", has_video=False, has_audio=True)
    assert "a caption" in tv and VIDEO_PLACEHOLDER in tv
    assert "a caption" in ta and AUDIO_PLACEHOLDER in ta


# --------------------------------------------------------------------------- #
# Batch routing                                                               #
# --------------------------------------------------------------------------- #


def test_text_batch_routes_without_media_inputs():
    wrapper = make_wrapper()
    out = wrapper._encode_batch({"text": ["one", "two"]})
    call = wrapper.model.calls[-1]
    assert out.shape == (2, OmniRetrieverWrapper.EMBED_DIM)
    assert "input_features" not in call
    assert "pixel_values_videos" not in call
    assert "input_raw_wav" not in call


def test_audio_batch_passes_raw_waveforms_for_beats():
    """BEATs consumes input_raw_wav; without it the audio adaptors are unused."""
    wrapper = make_wrapper()
    wrapper._encode_batch({"audio": [audio_item(), audio_item()]})
    call = wrapper.model.calls[-1]
    assert "input_features" in call
    assert len(call["input_raw_wav"]) == 2
    expected = int(8.0 * OmniRetrieverWrapper.AUDIO_SAMPLING_RATE)
    assert call["input_raw_wav"][0].shape == (expected,)


def test_audio_placeholders_are_doubled_for_beats():
    """Whisper and BEATs features interleave 1:1, so placeholders must double."""
    wrapper = make_wrapper()
    wrapper._encode_batch({"audio": [audio_item()]})
    prompt = wrapper.tokenizer.calls[-1]["prompts"][0]
    assert AUDIO_PLACEHOLDER * 2 in prompt
    assert prompt.count(AUDIO_PLACEHOLDER) % 2 == 0


def test_text_batch_does_not_double_anything():
    wrapper = make_wrapper()
    wrapper._encode_batch({"text": ["plain text"]})
    assert AUDIO_PLACEHOLDER not in wrapper.tokenizer.calls[-1]["prompts"][0]


def test_video_second_per_grid_is_passed_as_a_tensor():
    """Regression: the backbone computes ``2.0 * second_per_grids`` and indexes it
    per video. A plain list raises TypeError, which is what the stock processor
    avoids by wrapping outputs in BatchFeature(tensor_type="pt")."""
    wrapper = make_wrapper()
    wrapper._encode_batch({"video": [video_frames()], "video_duration": [8.0]})
    value = wrapper.model.calls[-1]["video_second_per_grid"]
    assert isinstance(value, torch.Tensor), f"got {type(value).__name__}"
    assert value.dtype.is_floating_point
    # the operations the backbone actually performs must not raise
    assert (2.0 * value).shape == value.shape
    assert float(value[0]) > 0


def test_video_batch_sets_second_per_grid_from_duration():
    wrapper = make_wrapper()
    wrapper._encode_batch({"video": [video_frames()], "video_duration": [4.0]})
    call = wrapper.processor.replace_calls[-1]
    # fps = 8 frames / 4 s = 2.0 -> second_per_grid = temporal_patch_size / fps = 1.0
    assert call["video_second_per_grid"] == [1.0]
    assert call["seconds_per_chunk"] == 2.0
    assert call["position_id_per_seconds"] == 25


def test_audio_video_batch_enables_use_audio_in_video():
    wrapper = make_wrapper()
    wrapper._encode_batch(
        {
            "video": [video_frames()],
            "video_duration": [8.0],
            "audio": [audio_item()],
        }
    )
    assert wrapper.processor.replace_calls[-1]["use_audio_in_video"] is True
    call = wrapper.model.calls[-1]
    assert call["use_audio_in_video"] is True
    assert "pixel_values_videos" in call and "input_features" in call


def test_single_modality_batches_do_not_enable_use_audio_in_video():
    wrapper = make_wrapper()
    wrapper._encode_batch({"audio": [audio_item()]})
    assert wrapper.processor.replace_calls[-1]["use_audio_in_video"] is False


def test_left_padding_is_used_for_last_token_pooling():
    """Pooling reads the final position, so right padding would pool padding."""
    wrapper = make_wrapper()
    wrapper._encode_batch({"text": ["short", "a much longer piece of text"]})
    assert wrapper.tokenizer.calls[-1]["padding_side"] == "left"


def test_pred_embeds_is_requested():
    """Without pred_embeds the backbone falls through to the contrastive loss path."""
    wrapper = make_wrapper()
    wrapper._encode_batch({"text": ["x"]})
    assert wrapper.model.calls[-1]["pred_embeds"] is True


# --------------------------------------------------------------------------- #
# Output contract                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "batch",
    [
        {"text": ["a", "b"]},
        {"audio": [audio_item(), audio_item()]},
        {"video": [video_frames(), video_frames()], "video_duration": [8.0, 8.0]},
        {
            "video": [video_frames(), video_frames()],
            "video_duration": [8.0, 8.0],
            "audio": [audio_item(), audio_item()],
        },
    ],
    ids=["text", "audio", "video", "audio_video"],
)
def test_every_modality_yields_finite_unit_norm_embeddings(batch):
    wrapper = make_wrapper()
    out = wrapper._encode_batch(batch)
    assert out.shape == (2, OmniRetrieverWrapper.EMBED_DIM)
    assert torch.isfinite(out).all()
    norms = out.norm(p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_embeddings_are_float32_regardless_of_model_dtype():
    wrapper = make_wrapper()
    out = wrapper._encode_batch({"text": ["x"]})
    assert out.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_empty_batch_raises_clearly():
    wrapper = make_wrapper()
    with pytest.raises(ValueError, match="empty batch"):
        wrapper._encode_batch({})


def test_batch_with_only_unsupported_modality_raises_clearly():
    """Images are not supported by the released adapter and must not pass silently."""
    wrapper = make_wrapper()
    with pytest.raises(ValueError, match="no\\s+image-only path"):
        wrapper._encode_batch({"image": ["not-supported"]})


# --------------------------------------------------------------------------- #
# Processor resolution                                                        #
# --------------------------------------------------------------------------- #


def test_processor_is_resolved_via_config_auto_map(monkeypatch):
    """Regression: WAVE-7B ships no processor_config.json, so AutoProcessor returns
    a bare Qwen2TokenizerFast. The auto_map entry must be resolved explicitly."""
    import transformers
    import transformers.dynamic_module_utils as dmu

    class FakeConfig:
        auto_map = {"AutoProcessor": "processing_qwen2_5_omni.Qwen2_5OmniProcessor"}

    resolved = {}

    class FakeOmniProcessor(StubProcessor):
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            resolved["name"] = name
            resolved["revision"] = kwargs.get("revision")
            return cls()

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda c, *a, **k: FakeConfig()),
    )
    monkeypatch.setattr(
        dmu, "get_class_from_dynamic_module", lambda ref, name, **k: FakeOmniProcessor
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        classmethod(
            lambda c, *a, **k: pytest.fail("must not fall back to AutoProcessor")
        ),
    )

    proc = OmniRetrieverWrapper._load_processor(WAVE_BASE_MODEL, WAVE_BASE_REVISION)
    assert isinstance(proc, FakeOmniProcessor)
    assert hasattr(proc, "replace_multimodal_special_tokens")
    assert resolved == {"name": WAVE_BASE_MODEL, "revision": WAVE_BASE_REVISION}


def test_processor_fallback_rejects_a_plain_tokenizer(monkeypatch):
    """If no auto_map exists, a tokenizer-only result must fail loudly, not silently."""
    import transformers

    class NoAutoMapConfig:
        auto_map = None

    class PlainTokenizer:  # no replace_multimodal_special_tokens
        pass

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda c, *a, **k: NoAutoMapConfig()),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        classmethod(lambda c, *a, **k: PlainTokenizer()),
    )

    with pytest.raises(TypeError, match="Qwen2.5-Omni style processor"):
        OmniRetrieverWrapper._load_processor(WAVE_BASE_MODEL, WAVE_BASE_REVISION)
