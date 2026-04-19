from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb._create_dataloaders import VideoCollator
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerMultimodalEncoderWrapper,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class OmniEmbedNemotronWrapper(SentenceTransformerMultimodalEncoderWrapper):
    """Thin wrapper that configures video/audio processing kwargs after loading."""

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length: int = 2_048_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, revision=revision, device=device, **kwargs)
        self.max_audio_length = max_audio_length
        self.sampling_rate = self.model[0].processor.feature_extractor.sampling_rate
        self.model[0].processing_kwargs.update(
            {
                "video": {
                    "min_pixels": 32 * 14 * 14,
                    "max_pixels": 64 * 28 * 28,
                    "do_sample_frames": False,
                },
                "audio": {"max_length": max_audio_length},
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
        if "video" in inputs.dataset.features or "audio" in inputs.dataset.features:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_audio_length,
            )
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


_OMNI_EMBED_NEMOTRON_CITATION = r"""
@article{xu2025omni,
    title={Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video},
    author={Xu, Mengyao and Zhou, Wenfei and Babakhin, Yauhen and Moreira, Gabriel and Ak, Ronay and Osmulski, Radek and Liu, Bo and Oldridge, Even and Schifferer, Benedikt},
    journal={arXiv preprint arXiv:2510.03458},
    year={2025}
}
"""

omni_embed_nemotron_3b = ModelMeta(
    loader=OmniEmbedNemotronWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="nvidia/omni-embed-nemotron-3b",
    revision="e0e93aaaa65d2422a8a0c1284116e71f7a0fe966",
    release_date="2025-10-01",
    languages=["eng-Latn"],
    n_parameters=4_703_464_448,
    memory_usage_mb=8971,
    max_tokens=32768,
    embed_dim=2048,
    n_embedding_parameters=311_164_928,
    license="https://huggingface.co/nvidia/omni-embed-nemotron-3b/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/omni-embed-nemotron-3b#training-dataset",
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/omni-embed-nemotron-3b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={
        "HotpotQA",
        "MIRACLRetrieval",
        "NQ",
        "VidoreArxivQARetrieval",
        "VidoreDocVQARetrieval",
        "VidoreInfoVQARetrieval",
        "VidoreTabfquadRetrieval",
        "VidoreTatdqaRetrieval",
        "VidoreShiftProjectRetrieval",
        "VidoreSyntheticDocQAAIRetrieval",
        "VidoreSyntheticDocQAEnergyRetrieval",
        "VidoreSyntheticDocQAGovernmentReportsRetrieval",
        "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        # "SQuAD",  # not directly in MTEB as retrieval task
        # "Stack Exchange",  # not directly in MTEB as retrieval task
        # "DocMatix-IR",  # not in MTEB
        # "Wiki-SS-NQ",  # not in MTEB
    },
    adapted_from="Qwen/Qwen2.5-Omni-3B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_OMNI_EMBED_NEMOTRON_CITATION,
    extra_requirements_groups=["multimodal_sbert"],
)
