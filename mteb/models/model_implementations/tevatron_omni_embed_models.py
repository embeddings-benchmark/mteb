from __future__ import annotations

from typing import Any

from mteb.models.model_implementations.colpali_models import COLPALI_TRAINING_DATA
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerMultimodalEncoderWrapper,
)


class TevatronOmniEmbedWrapper(SentenceTransformerMultimodalEncoderWrapper):
    """Thin wrapper that configures video processing kwargs after loading."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 2.0,
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
                    "max_pixels": 64
                    * 28
                    * 28,  # model card recommendation to save memory
                    "do_sample_frames": False,
                },
            }
        )


_OMNI_EMBED_CITATION = r"""
@article{zhuang2025tevatron,
    title={Tevatron 2.0: Unified Document Retrieval Toolkit across Scale, Language, and Modality},
    author={Zhuang, Shengyao and Ma, Xueguang and Zhan, Samantha and Zhang, Crystina},
    journal={arXiv preprint arXiv:2505.02466},
    year={2025}
}
"""

omni_embed_v01 = ModelMeta(
    loader=TevatronOmniEmbedWrapper,
    loader_kwargs={
        "trust_remote_code": True,
        "model_kwargs": {
            "torch_dtype": "bfloat16",
        },
    },
    name="Tevatron/OmniEmbed-v0.1",
    revision="fa8be315e74d07f563b07c0d053525de6dc7eda8",
    release_date="2025-04-12",
    languages=["eng-Latn"],
    n_parameters=8_931_813_888,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=17_036,
    max_tokens=32768,
    embed_dim=3584,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/texttron/tevatron/tree/qwenomni",
    public_training_data="https://huggingface.co/Tevatron/OmniEmbed-v0.1#training-data",
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Tevatron/OmniEmbed-v0.1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={
        "NQ",
        "MSMARCO",
        "FEVER",
        "SCIDOCS",
        "MSRVTTV2T",
        "MSRVTTT2V",
        "AudioCapsT2ARetrieval",
        *COLPALI_TRAINING_DATA,
        # "WikiSS-NQ",  # not in MTEB
        # "PixMo-Docs",  # not in MTEB
    },
    adapted_from="Qwen/Qwen2.5-Omni-7B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_OMNI_EMBED_CITATION,
    extra_requirements_groups=["multimodal-sbert", "peft"],
)
