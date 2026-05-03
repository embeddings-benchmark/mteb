from __future__ import annotations

from typing import Any

from transformers import AutoProcessor

from mteb.models.model_implementations.ops_colqwen3_models import (
    OpsColQwen3Wrapper,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction


class ArgusColQwen35Wrapper(OpsColQwen3Wrapper):
    """MTEB encoder for Argus-Colqwen3.5.

    Argus is a region-aware query-conditioned mixture-of-experts retriever
    built on the Qwen3.5-VL backbone. Identical to ``OpsColQwen3Wrapper``
    except that Argus's forward returns ``ArgusOutput(embeddings=...)``
    rather than a bare tensor, and the processor needs an explicit
    ``max_num_visual_tokens`` override.
    """

    def __init__(
        self,
        model_name: str = "DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
        revision: str | None = None,
        trust_remote_code: bool = True,
        max_num_visual_tokens: int = 2048,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        # Argus's processor accepts a configurable visual-token budget; the
        # base Ops processor does not pass it through, so re-build here.
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            max_num_visual_tokens=max_num_visual_tokens,
        )

    def encode_input(self, inputs):
        # Argus returns ``ArgusOutput(embeddings=...)`` instead of a tensor.
        return self.mdl(**inputs).embeddings


ARGUS_TRAINING_DATA = {
    # ViDoRe train-set subsets used during distillation. Reported here so the
    # MTEB leaderboard correctly flags any test-set overlap.
    "VDRMultilingualRetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
}

ARGUS_CITATION = """
@misc{argus2026,
  title  = {Argus: Region-Aware Query-Conditioned Mixture of Experts for Visual Document Retrieval},
  author = {DataScience-UIBK team},
  year   = {2026},
  url    = {https://huggingface.co/DataScience-UIBK/Argus-Colqwen3.5-4b-v0},
}"""


argus_colqwen35_4b = ModelMeta(
    loader=ArgusColQwen35Wrapper,
    name="DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
    loader_kwargs=dict(
        max_num_visual_tokens=2048,
        trust_remote_code=True,
    ),
    languages=["eng-Latn"],
    revision="fedffec17bc28034ce77f3e99500c6864c4d4b6b",
    release_date="2026-04-29",
    modalities=["image", "text"],
    n_parameters=4_708_446_726,
    n_embedding_parameters=635699200,
    memory_usage_mb=8981,
    max_tokens=32768,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=ARGUS_TRAINING_DATA,
    citation=ARGUS_CITATION,
    model_type=["late-interaction"],
)


argus_colqwen35_4b_bf16 = ModelMeta(
    loader=ArgusColQwen35Wrapper,
    name="DataScience-UIBK/Argus-Colqwen3.5-4b-v0-bf16",
    loader_kwargs=dict(
        max_num_visual_tokens=2048,
        trust_remote_code=True,
    ),
    languages=["eng-Latn"],
    revision="c88506c1bd05eb31ccf6a5c9b062bce0e8362520",
    release_date="2026-05-03",
    modalities=["image", "text"],
    n_parameters=4_708_446_726,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8981,
    max_tokens=32768,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/DataScience-UIBK/Argus-Colqwen3.5-4b-v0-bf16",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=ARGUS_TRAINING_DATA,
    citation=ARGUS_CITATION,
    model_type=["late-interaction"],
    adapted_from="DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
)
