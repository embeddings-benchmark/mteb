"""BasinRAG model definition for MTEB."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import SearchProtocol
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

BASINRAG_CITATION = """@software{martins2026basinrag,
  author = {Alex Martins},
  title = {BasinRAG: High-Performance Topological Retrieval-Augmented Generation},
  url = {https://github.com/Basinfy/BasinRAG},
  version = {1.0.3},
  year = {2026}
}"""


def basinrag_loader(
    model_name: str = "Basinfy/BasinRAG",
    revision: str | None = None,
    search_type: str = "hybrid",
    **kwargs: Any,
) -> SearchProtocol:
    """Loader function that instantiates BasinRAG as a SearchProtocol model for MTEB."""
    try:
        from basinrag.factory import BasinRAG
        from basinrag.indexer.condensation import node_layers
    except ImportError:
        raise ImportError(
            "BasinRAG is required to run this model. "
            "Please install it with: `pip install git+https://github.com/Basinfy/BasinRAG.git`"
        ) from None

    class BasinRAGSearchWrapper:
        """SearchProtocol wrapper for BasinRAG."""

        def __init__(
            self,
            search_type: str = "hybrid",
            storage_dir: str | None = None,
            **wrapper_kwargs: Any,
        ):
            import tempfile

            self.storage_dir = storage_dir or tempfile.mkdtemp(prefix="basinrag_mteb_")
            self.rag = BasinRAG.create(storage_dir=self.storage_dir)
            self.search_type = search_type

        def index(
            self,
            corpus: CorpusDatasetType,
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            encode_kwargs: EncodeKwargs,
            num_proc: int | None = None,
            **index_kwargs: Any,
        ) -> None:
            if isinstance(corpus, dict):
                doc_ids = list(corpus.keys())
                doc_texts = [
                    f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
                    if isinstance(corpus[did], dict)
                    else str(corpus[did])
                    for did in doc_ids
                ]
            else:
                doc_ids = list(
                    corpus["id"] if "id" in corpus.column_names else corpus["_id"]
                )
                titles = (
                    list(corpus["title"])
                    if "title" in corpus.column_names
                    else [""] * len(doc_ids)
                )
                texts = (
                    list(corpus["text"])
                    if "text" in corpus.column_names
                    else [""] * len(doc_ids)
                )
                doc_texts = [
                    f"{t} {x}".strip() for t, x in zip(titles, texts, strict=True)
                ]

            batch_size = 64
            all_embeddings = self.rag.ingestor.encoder.encode(
                doc_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            nodes = []
            for doc_id, text, emb in zip(
                doc_ids, doc_texts, all_embeddings, strict=True
            ):
                layers = node_layers(text)
                nodes.append(
                    {
                        "id": str(doc_id),
                        "text": text,
                        "embedding": emb,
                        "source": str(doc_id),
                        "chunk_index": 0,
                        "l1": layers["l1"],
                        "l2": layers["l2"],
                        "metadata": {"doc_id": str(doc_id), "id": str(doc_id)},
                    }
                )

            self.rag.engine.encoder_model = self.rag.config.encoder_model
            self.rag.engine.build_graph(nodes)
            self.rag.engine.partition_into_basins()
            self.rag.engine.build_meta_basins()
            self.rag._attach_bm25()
            self.rag.retriever = None

        def search(
            self,
            queries: QueryDatasetType,
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            top_k: int,
            encode_kwargs: EncodeKwargs,
            top_ranked: TopRankedDocumentsType | None = None,
            num_proc: int | None = None,
            **search_kwargs: Any,
        ) -> RetrievalOutputType:
            if isinstance(queries, dict):
                query_dict = queries
            else:
                q_ids = list(
                    queries["id"] if "id" in queries.column_names else queries["_id"]
                )
                q_texts = list(queries["text"])
                query_dict = {
                    str(qid): text for qid, text in zip(q_ids, q_texts, strict=True)
                }

            results: dict[str, dict[str, float]] = {}
            for qid, raw_qtext in query_dict.items():
                qtext = (
                    " ".join(str(m) for m in raw_qtext)
                    if isinstance(raw_qtext, list)
                    else str(raw_qtext)
                )
                docs = self.rag.query(
                    qtext, search_type=self.search_type, top_k=max(20, top_k * 2)
                )
                doc_scores: dict[str, float] = {}
                for rank, d in enumerate(docs):
                    meta = getattr(d, "metadata", {}) or {}
                    found_id = str(
                        meta.get("doc_id")
                        or meta.get("id")
                        or meta.get("source")
                        or meta.get("node_id")
                        or ""
                    )
                    if found_id and found_id not in doc_scores:
                        score = float(getattr(d, "score", 0.0) or (1.0 / (rank + 1.0)))
                        doc_scores[found_id] = score
                    if len(doc_scores) >= top_k:
                        break
                results[str(qid)] = doc_scores

            return results

    return BasinRAGSearchWrapper(search_type=search_type, **kwargs)


basinrag = ModelMeta(
    loader=basinrag_loader,
    name="Basinfy/BasinRAG",
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
