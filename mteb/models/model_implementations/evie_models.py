from __future__ import annotations

import logging
from typing import Any

import torch

from mteb.models.model_meta import ModelMeta, ScoringFunction

from .colqwen_models import ColQwen3_5Wrapper

logger = logging.getLogger(__name__)


class EvieWrapper(ColQwen3_5Wrapper):
    """EVIE: ColQwen3.5 with the full-attention layers encoder-ized.

    colpali_engine builds ColQwen3_5 causal. EVIE was trained and released with
    bidirectional attention over the full-attention layers (the linear/recurrent
    layers are left alone), so the flag has to be flipped after loading or the
    embeddings are not the ones the checkpoint was trained to produce.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_num_visual_tokens: int = 1024,
        attn_implementation: str = "flash_attention_2",
        **kwargs: Any,
    ):
        from transformers.utils.import_utils import is_flash_attn_2_available

        # EVIE's reported numbers come from flash_attention_2, and the choice is
        # not free: under bf16 the reduction order differs enough from sdpa to
        # move a ViDoRe v3 domain by ~0.15 nDCG@10. Pin it rather than probing,
        # so the same code scores the same everywhere.
        if (
            attn_implementation == "flash_attention_2"
            and not is_flash_attn_2_available()
        ):
            raise ImportError(
                "EVIE is evaluated with flash_attention_2; install it with "
                "`pip install mteb[evie]`. Passing attn_implementation='sdpa' "
                "also works but shifts nDCG@10 by roughly 0.15 per domain."
            )
        kwargs["attn_implementation"] = attn_implementation

        super().__init__(
            model_name=model_name, revision=revision, device=device, **kwargs
        )

        enable = getattr(self.model, "enable_bidirectional_attention", None)
        if callable(enable):
            enable()
        else:
            self._enable_bidirectional_attention()

        # The released page budget is 16384 visual tokens; reported results use 1024.
        from colpali_engine.models import ColQwen3_5Processor

        self.processor = ColQwen3_5Processor.from_pretrained(
            model_name,
            revision=revision,
            max_num_visual_tokens=max_num_visual_tokens,
        )

    def _enable_bidirectional_attention(self) -> None:
        config = self.model.config
        for cfg in (config, getattr(config, "text_config", None)):
            if cfg is not None:
                cfg.is_causal = False
        for module in self.model.modules():
            if module.__class__.__name__ in {"Qwen3_5Attention", "Qwen3Attention"}:
                module.is_causal = False


EVIE_CITATION = """
@misc{tencent2026evie,
  title        = {EVIE: Evidence-Vector-Informed Embedding for Visual Document Retrieval},
  author       = {Wang, Zifei and Wen, Wei},
  year         = {2026},
  howpublished = {\\url{https://github.com/Tencent/EVIE}}
}
"""

EVIE_TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    # from https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-Synthetic-data
    "VisRAG-Ret-Train-Synthetic-data",
    # from https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data
    "VisRAG-Ret-Train-In-domain-data",
    # from https://huggingface.co/datasets/llamaindex/vdr-multilingual-train
    "VDRMultilingualRetrieval",
    # from https://huggingface.co/datasets/Metric-AI/tabfquad_train_set
    "VidoreTabfquadRetrieval",
}

_EVIE_LANGUAGES = [
    "eng-Latn",
    "fra-Latn",
    "deu-Latn",
    "spa-Latn",
    "ita-Latn",
    "por-Latn",
]

evie_8b = ModelMeta(
    loader=EvieWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="tencent/EVIE-8B",
    model_type=["late-interaction"],
    languages=_EVIE_LANGUAGES,
    revision="b3cbfc6cf6bf18fe7da480eacb0f70017d666fe9",
    release_date="2026-09-04",
    modalities=["image", "text"],
    n_parameters=8_409_476_336,
    n_embedding_parameters=1_017_118_720,  # vocab 248320 x hidden 4096
    memory_usage_mb=16819,
    max_tokens=262144,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/Tencent/EVIE",
    public_training_data=None,
    framework=["PyTorch", "ColPali", "Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/tencent/EVIE-8B",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=EVIE_TRAINING_DATA,
    citation=EVIE_CITATION,
    extra_requirements_groups=["evie"],
)

evie_4_5b = ModelMeta(
    loader=EvieWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="tencent/EVIE-4.5B",
    model_type=["late-interaction"],
    languages=_EVIE_LANGUAGES,
    revision="8aecfa955e5e7d56a251291942f6f0717badb238",
    release_date="2026-09-04",
    modalities=["image", "text"],
    n_parameters=4_544_510_464,
    n_embedding_parameters=635_699_200,  # vocab 248320 x hidden 2560, tied
    memory_usage_mb=9089,
    max_tokens=262144,
    # Prefix-MRL: one Linear(2560, 2048) whose leading slices are valid
    # truncations. Results are reported at the full 2048 width.
    embed_dim=[64, 128, 256, 512, 1024, 2048],
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/Tencent/EVIE",
    public_training_data=None,
    framework=["PyTorch", "ColPali", "Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/tencent/EVIE-4.5B",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=EVIE_TRAINING_DATA,
    citation=EVIE_CITATION,
    extra_requirements_groups=["evie"],
)
