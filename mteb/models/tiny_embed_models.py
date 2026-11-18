"""Model metadata for the tiny-embed family (MTEB submission, PR-1).

Place this file at ``mteb/models/tiny_embed_models.py`` in the mteb repository.

Notes
-----
* ``training_datasets`` uses the ``{task_name: [splits]}`` format required by
  ``ModelMeta``. Only the subsets whose corpora actually appear in our training data are
  declared. T2Ranking-family corpora (``T2Retrieval`` / ``T2Reranking``), ``DuRetrieval``,
  ``ATEC``, ``BQ``, ``LCQMC`` and ``STSB`` were removed from training data, so those tasks
  are zero-shot for these models.
* ``use_instructions=False``: both models are trained and evaluated with no query prefix.
* ``tiny-embed-61m`` outputs 768-d embeddings: a 512-d backbone plus a bias-free linear
  512->768 projection that is part of the trained model (shipped as the ST ``Dense``
  module), not a post-processing step.
"""

from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

# Both checkpoints ship a custom architecture as ``tiny_st_module.py`` inside the repo,
# so ``trust_remote_code=True`` must be forwarded to SentenceTransformer.
# NOTE: ``ModelMeta.load_model`` calls the loader **without** positional arguments, so
# ``model_name`` / ``revision`` must be bound via functools.partial (same pattern as the
# existing BGE/Qwen model metas).
REV_61M = "1209a7bac8d676b6291d84ee7bec32193ae72d06"
REV_27M = "6eaaa7eac273a248748fef7276553cc09cc25b6e"

tiny_embed_61m_loader = partial(
    sentence_transformers_loader,
    model_name="dynasty-ai/tiny-embed-61m",
    revision=REV_61M,
    trust_remote_code=True,
)
tiny_embed_27m_loader = partial(
    sentence_transformers_loader,
    model_name="dynasty-ai/tiny-embed-27m",
    revision=REV_27M,
    trust_remote_code=True,
)

TINY_EMBED_TRAINING_DATASETS = {
    "PAWSX": ["train"],
    "CmedqaRetrieval": ["train"],
    "CMedQAv1-reranking": ["train"],
    "CMedQAv2-reranking": ["train"],
}

tiny_embed_61m = ModelMeta(
    loader=tiny_embed_61m_loader,
    name="dynasty-ai/tiny-embed-61m",
    revision=REV_61M,
    release_date="2026-11-18",
    languages=["zho-Hans"],
    n_parameters=61818880,  # 61,425,664 backbone + 393,216 projection head
    memory_usage_mb=123,
    max_tokens=512.0,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=TINY_EMBED_TRAINING_DATASETS,
    adapted_from=None,
    superseded_by=None,
)

tiny_embed_27m = ModelMeta(
    loader=tiny_embed_27m_loader,
    name="dynasty-ai/tiny-embed-27m",
    revision=REV_27M,
    release_date="2026-11-18",
    languages=["zho-Hans"],
    n_parameters=27599872,
    memory_usage_mb=55,
    max_tokens=512.0,
    embed_dim=512,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=TINY_EMBED_TRAINING_DATASETS,
    adapted_from=None,
    superseded_by=None,
)
