from __future__ import annotations

from typing import Any

from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerMultimodalEncoderWrapper,
)


class E5OmniWrapper(SentenceTransformerMultimodalEncoderWrapper):
    """Thin wrapper that configures video processing kwargs after loading."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 1.0,
        max_frames: int | None = None,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            **kwargs,
        )
        self.target_sampling_rate = self.model[
            0
        ].processor.feature_extractor.sampling_rate
        self.model[0].processing_kwargs.update(
            {
                "video": {
                    "max_pixels": 64 * 28 * 28,  # model card recommendation
                    "do_sample_frames": False,
                },
            }
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
    extra_requirements_groups=["multimodal_sbert"],
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
    extra_requirements_groups=["multimodal_sbert"],
)
