from __future__ import annotations

from typing import Any

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)


class AuroLAOmniWrapper(SentenceTransformerEncoderWrapper):
    """Wrapper for AuroLA-Omni-7B omni-modal embedding model.

    Configures recommended video processing parameters and ensures Qwen2.5-Omni
    architecture registration with AutoModelForMultimodalLM.
    """

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 1.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_MULTIMODAL_LM_MAPPING,
                AutoModelForMultimodalLM,
            )
            from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
                Qwen2_5OmniThinkerConfig,
            )
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniThinkerForConditionalGeneration,
            )

            if (
                Qwen2_5OmniThinkerConfig not in MODEL_FOR_MULTIMODAL_LM_MAPPING
                and hasattr(AutoModelForMultimodalLM, "register")
            ):
                AutoModelForMultimodalLM.register(
                    Qwen2_5OmniThinkerConfig,
                    Qwen2_5OmniThinkerForConditionalGeneration,
                )
        except (ImportError, AttributeError):
            pass

        super().__init__(
            model,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            **kwargs,
        )
        if hasattr(self.model[0], "processor") and hasattr(
            self.model[0].processor, "feature_extractor"
        ):
            self.target_sampling_rate = self.model[
                0
            ].processor.feature_extractor.sampling_rate
        if hasattr(self.model[0], "processing_kwargs"):
            self.model[0].processing_kwargs.setdefault("video", {}).update(
                {
                    "max_pixels": 64 * 28 * 28,
                    "do_sample_frames": False,
                    "fps": 1,
                }
            )


AUROLA_TRAINING_DATASETS = {
    "AudioCapsA2TRetrieval",
    "AudioCapsT2ARetrieval",
    "ClothoA2TRetrieval",
    "ClothoT2ARetrieval",
}

_AUROLA_CITATION = r"""
@misc{xu2026scalingaudiotextretrievalmultimodal,
    title={Scaling Audio-Text Retrieval with Multimodal Large Language Models},
    author={Jilan Xu and Carl Thom{\'e} and Danijela Horak and Weidi Xie and Andrew Zisserman},
    year={2026},
    eprint={2602.18010},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
    url={https://arxiv.org/abs/2602.18010}
}
"""

aurola_omni_7b = ModelMeta(
    loader=AuroLAOmniWrapper,
    loader_kwargs={
        "trust_remote_code": True,
        "model_kwargs": {
            "torch_dtype": "bfloat16",
        },
    },
    name="Jazzcharles/AuroLA-Omni-7B",
    revision="414ec2ae4f35782a019fff87c80db6a19d347bc4",
    release_date="2026-08-23",
    languages=["eng-Latn"],
    n_parameters=8_928_961_024,
    n_embedding_parameters=543_570_944,
    memory_usage_mb=17_030,
    max_tokens=32768,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/Jazzcharles/AuroLA",
    public_training_data="https://github.com/Jazzcharles/AuroLA",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Jazzcharles/AuroLA-Omni-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=AUROLA_TRAINING_DATASETS,
    adapted_from="Qwen/Qwen2.5-Omni-7B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_AUROLA_CITATION,
    extra_requirements_groups=["qwen-vl"],
)

aurola_omni_3b = ModelMeta(
    loader=AuroLAOmniWrapper,
    loader_kwargs={
        "trust_remote_code": True,
        "model_kwargs": {
            "torch_dtype": "bfloat16",
        },
    },
    name="Jazzcharles/AuroLA-Omni-3B",
    revision="a1916873e5c12204f843ceea1f74e3aafa9bc93a",
    release_date="2026-08-23",
    languages=["eng-Latn"],
    n_parameters=4_702_358_528,
    n_embedding_parameters=310_611_968,
    memory_usage_mb=8_969,
    max_tokens=32768,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/Jazzcharles/AuroLA",
    public_training_data="https://github.com/Jazzcharles/AuroLA",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Jazzcharles/AuroLA-Omni-3B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=AUROLA_TRAINING_DATASETS,
    adapted_from="Qwen/Qwen2.5-Omni-3B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_AUROLA_CITATION,
    extra_requirements_groups=["qwen-vl"],
)
