"""BasinRAG model definition for MTEB."""
from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction

BASINRAG_CITATION = """@software{martins2026basinrag,
  author = {Alex Martins},
  title = {BasinRAG: High-Performance Topological Retrieval-Augmented Generation},
  url = {https://github.com/Basinfy/BasinRAG},
  version = {1.0.3},
  year = {2026}
}"""

basinrag = ModelMeta(
    loader=None,
    name="alexmart1ns/BasinRAG",
    model_type=["hybrid"],
    languages=["eng-Latn", "por-Latn"],
    open_weights=True,
    revision="1.0.3",
    release_date="2026-09-08",
    n_parameters=118_000_000,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=450,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://github.com/Basinfy/BasinRAG",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/Basinfy/BasinRAG",
    public_training_data=None,
    training_datasets=None,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    citation=BASINRAG_CITATION,
)
