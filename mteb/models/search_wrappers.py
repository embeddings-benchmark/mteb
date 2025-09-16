from __future__ import annotations

import heapq
import logging
from typing import Any

import torch
from datasets import Dataset

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import (
    create_dataloader,
)
from mteb.types import (
    Array,
    CorpusDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from .models_protocols import CrossEncoderProtocol, Encoder


logger = logging.getLogger(__name__)


class SearchEncoderWrapper:
    corpus_chunk_size = 50_000
    task_corpus: CorpusDatasetType | None
    _index_dir: str | None
    _index_name: str | None
    _index_autodelete: bool

    def __init__(self, model: Encoder):
        self.model = model
        self.task_corpus = None
        self.mteb_model_meta = model.mteb_model_meta

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        index_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            index_kwargs: Additional arguments to configure the search index (e.g. PyLate settings).
        """
        # Always retain corpus for potential reranking or fallback flows
        self.task_corpus = corpus

        # Reset index state so non-PyLate paths are untouched
        self._index_dir = None
        self._index_name = None
        self._index_autodelete = True
     
        # Compute and persist a generic temporary index path (backend-agnostic)
        is_pylate = bool(
            self.mteb_model_meta is not None
            and getattr(self.mteb_model_meta, "is_pylate_compatible", False) is True
        )

        import tempfile
        safe_task = task_metadata.name.replace("/", "_")

        index_dir = index_kwargs.pop("index_dir", None)
        if index_dir is None:
            index_dir = tempfile.mkdtemp(prefix=f"mteb-index-{safe_task}-{hf_subset}-{hf_split}-")

        index_name = index_kwargs.pop("index_name", None)
        if index_name is None:
            index_name = "index"

        index_autodelete = index_kwargs.pop("index_autodelete", None)
        if index_autodelete is None:
            index_autodelete = True

        self._index_dir = index_dir
        self._index_name = index_name
        self._index_autodelete = bool(index_autodelete)

        if is_pylate:
            self._index_multivector_pylate(
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                index_dir=self._index_dir,
                index_name=self._index_name,
                index_kwargs=index_kwargs,
            )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        index_kwargs: dict[str, Any] | None = None,
        top_ranked: TopRankedDocumentsType | None = None,
    ) -> RetrievalOutputType:
        """Search the corpus for the given queries.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
                Passed only from Reranking tasks.
            top_k: Number of top documents to return for each query.
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
            index_kwargs: Additional arguments to configure the search index during retrieval.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        queries_dataloader = create_dataloader(
            queries,
            task_metadata,
            prompt_type=PromptType.query,
            batch_size=encode_kwargs.get("batch_size", 32),
        )

        query_embeddings = self.model.encode(
            queries_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        query_idx_to_id = {i: row["id"] for i, row in enumerate(queries)}

        is_pylate = bool(
            self.mteb_model_meta is not None
            and getattr(self.mteb_model_meta, "is_pylate_compatible", False) is True
        )

        if top_ranked is not None:
            if is_pylate:
                logger.info("Reranking with PyLate...")
                result_heaps = self._pylate_rerank_documents(
                    query_idx_to_id=query_idx_to_id,
                    query_embeddings=query_embeddings,
                    top_ranked=top_ranked,
                    top_k=top_k,
                    task_metadata=task_metadata,
                    hf_subset=hf_subset,
                    hf_split=hf_split,
                    encode_kwargs=encode_kwargs,
                )
            else:
                logger.info("Reranking pre-ranked documents...")
                result_heaps = self._rerank_documents(
                    query_idx_to_id=query_idx_to_id,
                    query_embeddings=query_embeddings,
                    top_ranked=top_ranked,
                    top_k=top_k,
                    task_metadata=task_metadata,
                    hf_subset=hf_subset,
                    hf_split=hf_split,
                    encode_kwargs=encode_kwargs,
                )
        else:
            if is_pylate:
                logger.info("Retrieving with MultiVector index...")
                try:
                    from pylate import retrieve, indexes  # type: ignore
                except Exception as e:  # pragma: no cover - import error path
                    raise ImportError(
                        "PyLate is required for PyLate-compatible models. Install with `pip install mteb[pylate]`."
                    ) from e

                if self._index_dir is None or self._index_name is None:
                    raise ValueError("Index path is not set. Call index() before search().")

                index = indexes.PLAID(
                    index_folder=self._index_dir,
                    index_name=self._index_name,
                    **index_kwargs
                )
                retriever = retrieve.ColBERT(index=index)
                scores = retriever.retrieve(queries_embeddings=query_embeddings, k=top_k)

                # Build heaps in the same structure as dense path for consistency
                result_heaps = {qid: [] for qid in query_idx_to_id.values()}
                for q_idx, qid in query_idx_to_id.items():
                    # scores[q_idx] is a list of dicts: {"id": str, "score": float}
                    for item in scores[q_idx]:
                        heapq.heappush(result_heaps[qid], (float(item["score"]), str(item["id"])))
            else:
                logger.info("Performing full corpus search...")
                result_heaps = self._full_corpus_search(
                    query_idx_to_id=query_idx_to_id,
                    query_embeddings=query_embeddings,
                    task_metadata=task_metadata,
                    hf_subset=hf_subset,
                    hf_split=hf_split,
                    top_k=top_k,
                    encode_kwargs=encode_kwargs,
                )

        # Reset the task corpus dataloader to None to free up memory
        self.task_corpus = None
        # Optionally clean up the on-disk index
        if self._index_autodelete and self._index_dir is not None:
            import shutil
            try:
                shutil.rmtree(self._index_dir, ignore_errors=True)
            finally:
                self._index_dir = None
                self._index_name = None

        results = {qid: {} for qid in query_idx_to_id.values()}
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                results[qid][corpus_id] = score

        return results

    def _full_corpus_search(
        self,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        itr = range(0, len(self.task_corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size, len(self.task_corpus)
            )
            sub_corpus = self.task_corpus.select(
                range(corpus_start_idx, corpus_end_idx)
            )
            sub_corpus_ids = sub_corpus["id"]
            sub_corpus_embeddings = self.model.encode(
                create_dataloader(
                    sub_corpus,
                    task_metadata,
                    prompt_type=PromptType.document,
                    batch_size=encode_kwargs.get("batch_size", 32),
                ),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )

            # Compute similarities using either cosine-similarity or dot product
            logger.info("Computing Similarities...")
            scores = self.model.similarity(query_embeddings, sub_corpus_embeddings)

            # get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                scores,
                min(
                    top_k + 1,
                    len(scores[1]) if len(scores) > 1 else len(scores[-1]),
                ),
                dim=1,
                largest=True,
            )
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_idx_to_id[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr],
                    cos_scores_top_k_values[query_itr],
                ):
                    corpus_id = sub_corpus_ids[sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        # push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
        return result_heaps

    def _index_multivector_pylate(
        self,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        index_dir: str,
        index_name: str,
        index_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Build a MultiVector (PyLate) index in a backend-agnostic temp location."""
        try:
            from pylate import indexes  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise ImportError(
                "PyLate is required for PyLate-compatible models. Install with `pip install mteb[pylate]`."
            ) from e

        index_kwargs = index_kwargs or {}

        # Initialize index
        index = indexes.PLAID(
            index_folder=index_dir,
            index_name=index_name,
            **index_kwargs,
        )

        # Collect all IDs
        doc_ids = [str(x) for x in self.task_corpus["id"]]

        # Encode entire corpus via dataloader batching
        documents_loader = create_dataloader(
            self.task_corpus,
            task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        documents_embeddings = self.model.encode(
            documents_loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )

        # Add documents to index
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=documents_embeddings,
        )

    def _pylate_rerank_documents(
        self,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        top_ranked: TopRankedDocumentsType,
        top_k: int,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        encode_kwargs: dict[str, Any],
        **kwargs
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank with PyLate's rank.rerank using per-query candidates.

        Keeps dense rerank untouched by using a PyLate-only path.
        """
        try:
            from pylate import rank  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyLate is required for PyLate-compatible models. Install with `pip install mteb[pylate]`."
            ) from e

        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before reranking.")

        # Map doc_id -> dataset index
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}

        result_heaps = {qid: [] for qid in query_idx_to_id.values()}

        # Process one query at a time to keep it simple
        for q_idx, qid in query_idx_to_id.items():
            if qid not in top_ranked:
                continue
            ranked_ids = top_ranked[qid]
            if not ranked_ids:
                continue

            # Select candidate documents preserving order
            indices = [doc_id_to_idx[doc_id] for doc_id in ranked_ids if doc_id in doc_id_to_idx]
            if not indices:
                continue
            sub_corpus = self.task_corpus.select(indices)

            documents_loader = create_dataloader(
                sub_corpus,
                task_metadata,
                prompt_type=PromptType.document,
                batch_size=encode_kwargs.get("batch_size", 32),
            )
            documents_embeddings = self.model.encode(
                documents_loader,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )

            # Single-query rerank for simplicity
            q_emb = query_embeddings[q_idx:q_idx+1]
            reranked = rank.rerank(
                documents_ids=[ranked_ids],
                queries_embeddings=q_emb,
                documents_embeddings=[documents_embeddings],
            )

            # Parse PyLate's output
            for item in reranked[0]:  # list of dicts
                heapq.heappush(result_heaps[qid], (float(item["score"]), str(item["id"])))

            # Keep only top_k
            if len(result_heaps[qid]) > top_k:
                result_heaps[qid] = heapq.nlargest(top_k, result_heaps[qid])

        return result_heaps

    def _rerank_documents(
        self,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        top_ranked: TopRankedDocumentsType,
        top_k: int,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank documents based on pre-ranked documents."""
        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}

        all_doc_embeddings = self.model.encode(
            create_dataloader(
                self.task_corpus,
                task_metadata,
                prompt_type=PromptType.document,
                batch_size=encode_kwargs.get("batch_size", 32),
            ),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )

        # Process each query
        for query_idx, query_embedding in enumerate(query_embeddings):
            query_id = query_idx_to_id[query_idx]
            if query_id not in top_ranked:
                logger.warning(f"No pre-ranked documents found for query {query_id}")
                continue

            ranked_ids = top_ranked[query_id]
            doc_indices = torch.tensor([doc_id_to_idx[doc_id] for doc_id in ranked_ids])
            query_doc_embeddings = torch.as_tensor(all_doc_embeddings[doc_indices])

            # Ensure query embedding is on the correct device and has correct shape
            query_embedding = torch.as_tensor(query_embedding).unsqueeze(0)

            scores = self.model.similarity(
                query_embedding,
                query_doc_embeddings,
            )

            # Handle NaN values
            is_nan = torch.isnan(scores)
            if is_nan.sum() > 0:
                raise ValueError(
                    f"NaN values detected in the similarity scores: {is_nan.sum()}"
                )

            # Compute top-k scores
            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores,
                min(top_k, len(ranked_ids)),
                dim=1,
                largest=True,
            )

            # Move results back to CPU for heap operations
            scores_top_k_values = scores_top_k_values.cpu()
            scores_top_k_idx = scores_top_k_idx.cpu()

            # Build result heap
            for doc_idx, score in zip(
                scores_top_k_idx[0].tolist(),
                scores_top_k_values[0].tolist(),
            ):
                corpus_id = ranked_ids[doc_idx]
                heapq.heappush(result_heaps[query_id], (score, corpus_id))

        return result_heaps


class SearchCrossEncoderWrapper:
    task_corpus: CorpusDatasetType | None

    def __init__(self, model: CrossEncoderProtocol):
        self.model = model
        self.task_corpus = None
        self.mteb_model_meta = model.mteb_model_meta

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        index_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.task_corpus = corpus

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: TopRankedDocumentsType | None = None,
    ) -> RetrievalOutputType:
        if top_ranked is None:
            raise ValueError(
                "CrossEncoder search requires top_ranked documents for reranking."
            )

        query_id_to_idx = {row["id"]: i for i, row in enumerate(queries)}
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}

        total_queries = []
        total_docs = []
        doc_pairs_ids: list[tuple[str, str]] = []
        for query_id, corpus_ids in top_ranked.items():
            if query_id not in top_ranked:
                logger.warning(f"No pre-ranked documents found for query {query_id}")
                continue

            query_idx = query_id_to_idx[query_id]
            for corpus_id in corpus_ids:
                doc_pairs_ids.append((query_id, corpus_id))
                total_queries.append(queries[query_idx])
                total_docs.append(self.task_corpus[doc_id_to_idx[corpus_id]])

        queries_loader = create_dataloader(
            Dataset.from_list(total_queries),
            task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        corpus_loader = create_dataloader(
            Dataset.from_list(total_docs),
            task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        predictions = self.model.predict(
            inputs1=queries_loader,
            inputs2=corpus_loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )

        results = {qid: {} for qid in queries["id"]}
        for (query_id, corpus_id), score in zip(doc_pairs_ids, predictions):
            results[query_id][corpus_id] = float(score)

        return results
