from __future__ import annotations

import logging
from typing import Any

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import OutputDType

from .colqwen_models import ColQwen3_5Wrapper

logger = logging.getLogger(__name__)


def _enable_bidirectional_attention(model: Any) -> None:  # ruff: ignore[any-type]
    """Encoder-ize the full-attention layers of a ColQwen3.5 backbone.

    Mirrors `ColQwen3_5.enable_bidirectional_attention` from the EVIE release:
    https://github.com/Tencent/EVIE/blob/main/colpali/colpali_engine/models/qwen3_5/colqwen3_5/modeling_colqwen3_5.py#L20

    Two switches have to be flipped, and they are not interchangeable:

    * `config.is_causal=False` is what `create_causal_mask` reads to build a
      bidirectional mask instead of a causal one. Whenever a batch is padded
      the mask tensor is what sdpa actually honours, so without this the
      attention silently stays causal.
    * `Qwen3_5Attention.is_causal=False` is what flash_attention_2 reads, since
      FA2 only receives the 2D padding mask.

    Qwen3.5 interleaves GatedDeltaNet (`linear_attention`) and dense
    (`full_attention`) layers; only the dense ones are touched.
    """
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.is_causal = False

    text_model = model.language_model
    layer_types = text_model.config.layer_types
    for layer, layer_type in zip(text_model.layers, layer_types, strict=True):
        if layer_type == "full_attention":
            layer.self_attn.is_causal = False


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
        # move a ViDoRe v3 domain by ~0.15 nDCG@10. Prefer FA2 when it is
        # available, but fall back to sdpa so the model still runs without it.
        if (
            attn_implementation == "flash_attention_2"
            and not is_flash_attn_2_available()
        ):
            logger.warning(
                "flash_attention_2 is not available; falling back to "
                "attn_implementation='sdpa'. EVIE's reported nDCG@10 uses "
                "flash_attention_2, and sdpa can shift it by roughly 0.15 per "
                "domain. Install it with `pip install flash-attn` to reproduce "
                "the reported numbers exactly."
            )
            attn_implementation = "sdpa"
        kwargs["attn_implementation"] = attn_implementation

        super().__init__(
            model_name=model_name, revision=revision, device=device, **kwargs
        )

        _enable_bidirectional_attention(self.model)

        # The released page budget is 16384 visual tokens; reported results use 1024.
        from colpali_engine.models import ColQwen3_5Processor

        self.processor = ColQwen3_5Processor.from_pretrained(
            model_name,
            revision=revision,
            max_num_visual_tokens=max_num_visual_tokens,
        )


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
        torch_dtype=OutputDType.BF16,
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
        torch_dtype=OutputDType.BF16,
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
