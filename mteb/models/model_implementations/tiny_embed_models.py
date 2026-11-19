"""Model metadata for the tiny-embed family (MTEB submission).

Place at ``mteb/models/model_implementations/tiny_embed_models.py``.

Both checkpoints ship a custom architecture as ``tiny_st_module.py`` inside the Hub repo,
so ``trust_remote_code=True`` is forwarded through ``loader_kwargs``. ``load_model()``
passes ``name`` and ``revision=...`` to the loader itself, so they do not need to be bound.

``training_datasets`` lists the MTEB tasks whose corpora appear in our training data.
T2Ranking-family corpora (``T2Retrieval`` / ``T2Reranking``), ``DuRetrieval``, ``ATEC``,
``BQ``, ``LCQMC`` and ``STSB`` were removed from training data, so those tasks are
zero-shot for these models.

``tiny-embed-61m`` outputs 768-d embeddings: a 512-d backbone plus a bias-free linear
512->768 projection that is part of the trained model (shipped as the ST ``Dense``
module), not a post-processing step.
"""

from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

TINY_EMBED_TRAINING_DATASETS = {
    "PAWSX",
    "CmedqaRetrieval",
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
}

tiny_embed_61m = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="dynasty-ai/tiny-embed-61m",
    revision="1209a7bac8d676b6291d84ee7bec32193ae72d06",
    release_date="2026-11-18",
    languages=["zho-Hans"],
    n_parameters=61_818_880,  # 61,425,664 backbone + 393,216 projection head
    n_embedding_parameters=10_817_536,  # token 21,128x512 (mteb convention excludes position)
    memory_usage_mb=236,  # fp32 checkpoint (61.8M params)
    max_tokens=512.0,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/dynasty-ai/tiny-embed-61m",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=TINY_EMBED_TRAINING_DATASETS,
    adapted_from=None,
    superseded_by=None,
)

tiny_embed_27m = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="dynasty-ai/tiny-embed-27m",
    revision="6eaaa7eac273a248748fef7276553cc09cc25b6e",
    release_date="2026-11-18",
    languages=["zho-Hans"],
    n_parameters=27_599_872,
    n_embedding_parameters=10_817_536,  # token 21,128x512 (RoPE, no learned position embedding)
    memory_usage_mb=105,  # fp32 checkpoint (27.6M params)
    max_tokens=512.0,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/dynasty-ai/tiny-embed-27m",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=TINY_EMBED_TRAINING_DATASETS,
    adapted_from=None,
    superseded_by=None,
)
