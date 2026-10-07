from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.models.modality_collators import check_duration_cap
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class E5OmniWrapper(SentenceTransformerEncoderWrapper):
    """Thin wrapper that configures video processing kwargs after loading."""

    # batched audio embeddings drift far from single-clip ones on real clips,
    # and padded batches of long clips use up to ~60 GB
    audio_one_clip_reason = "batched audio embeddings change with their batch-mates"

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        # fps=2: qwen-omni-utils FPS=2.0
        # https://github.com/QwenLM/Qwen2.5-Omni/blob/main/qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py
        fps: float | None = 2.0,
        # 64 is an mteb cap; upstream ships FPS_MAX_FRAMES=768
        max_frames: int | None = 64,
        num_frames: int | None = None,
        # 300 s at 16 kHz: chunk_length=300
        # https://huggingface.co/Haon-Chen/e5-omni-3B/blob/main/preprocessor_config.json
        max_samples: int | None = 4_800_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            max_samples=check_duration_cap(max_samples),
            **kwargs,
        )
        self.target_sampling_rate = self.model[
            0
        ].processor.feature_extractor.sampling_rate
        self.model[0].processing_kwargs.update(
            {
                "video": {
                    "min_pixels": 32 * 14 * 14,
                    "max_pixels": 64 * 28 * 28,
                    "do_sample_frames": False,
                },
                "text": {"truncation": True, "max_length": 512},
            }
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        # the 512-token limit is for text; audio and video become tokens in the
        # same sequence, so truncating there would cut them (~20 s of audio)
        text_only = not any(
            m in inputs.dataset.features for m in ("image", "audio", "video")
        )
        self.model[0].processing_kwargs["text"] = (
            {"truncation": True, "max_length": 512} if text_only else {}
        )
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


_E5_OMNI_CITATION = r"""
@article{chen2026e5omni,
    title={e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings},
    author={Chen, Haonan and Gao, Sicheng and Radu, Timofte and Tetsuya, Sakai and Dou, Zhicheng},
    journal={arXiv preprint arXiv:2601.03666},
    year={2026}
}
"""

e5_omni_3b = ModelMeta(
    loader=E5OmniWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="Haon-Chen/e5-omni-3B",
    revision="bc2c24d7596ea578d08adffd96146ed47b1e5f72",
    release_date="2026-01-06",
    languages=["eng-Latn"],
    n_parameters=4_703_464_448,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=8_971,
    max_tokens=32768,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Haon-Chen/e5-omni-3B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={
        "MSRVTTV2T",
        "MSRVTTT2V",
        "AudioCapsT2ARetrieval",
        # "BGE-m3",  # not directly in MTEB
        # "MMEB-V1",  # not directly in MTEB
        # "MMEB-V2",  # not directly in MTEB
        # "PixMo-Docs",  # not in MTEB
    },
    adapted_from="Qwen/Qwen2.5-Omni-3B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_E5_OMNI_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

e5_omni_7b = ModelMeta(
    loader=E5OmniWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="Haon-Chen/e5-omni-7B",
    revision="ffea4ae1382fc26dc9fc337a89ced3fab58e408b",
    release_date="2026-01-06",
    languages=["eng-Latn"],
    n_parameters=8_931_813_888,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=17_036,
    max_tokens=32768,
    embed_dim=3584,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Haon-Chen/e5-omni-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={
        "MSRVTTV2T",
        "MSRVTTT2V",
        "AudioCapsT2ARetrieval",
        # "BGE-m3",  # not directly in MTEB
        # "MMEB-V1",  # not directly in MTEB
        # "MMEB-V2",  # not directly in MTEB
        # "PixMo-Docs",  # not in MTEB
    },
    adapted_from="Qwen/Qwen2.5-Omni-7B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_E5_OMNI_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)
