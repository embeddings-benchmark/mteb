"""GreenLeaf Law Embed models by JudicialMind.

Custom embedding models purpose-built for legal text retrieval.
Uses bidirectional attention on Qwen3 architecture with custom code
for flexible quantization (bf16, int8, binary).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class GreenLeafEmbedWrapper(SentenceTransformerEncoderWrapper):
    """Wrapper for GreenLeaf Law Embed models.

    These models use trust_remote_code=True for custom bidirectional
    attention and flexible quantization support.
    """

    def __init__(self, model_name: str, revision: str | None = None, **kwargs):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        super().__init__(model=model, **kwargs)


greenleaf_law_embed_tiny = ModelMeta(
    loader=GreenLeafEmbedWrapper,
    loader_kwargs={},
    name="judicialmind/greenleaf-law-embed-tiny",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="bff06c9474bda26e7b5883640821d6910689eb82",
    release_date="2026-08-22",
    n_parameters=595_776_512,
    n_embedding_parameters=155_189_248,
    memory_usage_mb=1136,
    max_tokens=32768,
    embed_dim=1024,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=False,
    reference="https://huggingface.co/judicialmind/greenleaf-law-embed-tiny",
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
