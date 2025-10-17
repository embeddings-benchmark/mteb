import heapq
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb._create_dataloaders import (
    create_dataloader,
)
from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import (
    Array,
    BatchedInput,
    CorpusDatasetType,
    PromptType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

logger = logging.getLogger(__name__)


class PylateSearchEncoder:
    """Mixin class to add PyLate-based indexing and search to an encoder. Implements :class:`SearchProtocol`"""

    base_index_dir: Path | None = None
    _index_dir: Path | None = None
    _index_name: str | None = None
    _index_autodelete: bool = True
    task_corpus: CorpusDatasetType | None = None
    index_kwargs: dict[str, Any] = {}  # noqa: RUF012

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encode_kwargs: Additional arguments to pass to the encoder during indexing.
        """
        self.task_corpus = corpus

        safe_task = task_metadata.name.replace("/", "__")

        index_dir_name = f"mteb-index-{safe_task}-{hf_subset}-{hf_split}"
        if self.base_index_dir is None:
            self._index_dir = Path(tempfile.mkdtemp(prefix=index_dir_name))
        else:
            self._index_dir = self.base_index_dir / index_dir_name
            self._index_dir.mkdir(parents=True, exist_ok=True)

        if self._index_name is None:
            self._index_name = "index"

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
        queries_dataloader = create_dataloader(
            queries,
            task_metadata,
            prompt_type=PromptType.query,
            batch_size=encode_kwargs.get("batch_size", 32),
        )

        query_embeddings = self.encode(
            queries_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        query_idx_to_id = {i: row["id"] for i, row in enumerate(queries)}

        if top_ranked is not None:
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
            result_heaps = self._pylate_full_corpus_search(
                query_idx_to_id=query_idx_to_id,
                query_embeddings=query_embeddings,
                top_k=top_k,
                task_metadata=task_metadata,
                hf_subset=hf_subset,
                hf_split=hf_split,
                encode_kwargs=encode_kwargs,
            )

        results = {qid: {} for qid in query_idx_to_id.values()}
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                results[qid][corpus_id] = score

        return results

    def _pylate_full_corpus_search(
        self,
        query_idx_to_id: dict[int, str],
        query_embeddings: Array,
        task_metadata: TaskMetadata,
        hf_subset: str,
        hf_split: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
    ) -> dict[str, list[tuple[float, str]]]:
        from pylate import indexes, retrieve

        logger.info("Retrieving with MultiVector index...")

        if self._index_dir is None or self._index_name is None:
            raise ValueError("Index path is not set. Call index() before search().")

        index = indexes.PLAID(
            index_folder=str(self._index_dir),
            index_name=self._index_name,
            # disable triton kernel for reproducibility
            # https://github.com/embeddings-benchmark/mteb/pull/3183#issuecomment-3311029707
            use_triton=False,
            **self.index_kwargs,
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
        documents_embeddings = self.encode(
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
        retriever = retrieve.ColBERT(index=index)
        scores = retriever.retrieve(queries_embeddings=query_embeddings, k=top_k)

        # Build heaps in the same structure as dense path for consistency
        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        for q_idx, qid in query_idx_to_id.items():
            # scores[q_idx] is a list of dicts: {"id": str, "score": float}
            for item in scores[q_idx]:
                heapq.heappush(
                    result_heaps[qid], (float(item["score"]), str(item["id"]))
                )
        return result_heaps

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
    ) -> dict[str, list[tuple[float, str]]]:
        """Rerank with PyLate's rank.rerank using per-query candidates.

        Keeps dense rerank untouched by using a PyLate-only path.

        Returns:
            A dictionary mapping query IDs to a list of tuples, each containing a score and a document ID.
        """
        from pylate import rank

        if self.task_corpus is None:
            raise ValueError("Corpus must be set before reranking.")

        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}

        all_doc_embeddings = self.encode(
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

        # Process one query at a time to keep it simple
        for q_idx, qid in query_idx_to_id.items():
            if qid not in top_ranked:
                continue
            ranked_ids = top_ranked[qid]
            if not ranked_ids:
                continue

            doc_indices = torch.tensor([doc_id_to_idx[doc_id] for doc_id in ranked_ids])
            query_doc_embeddings = torch.as_tensor(all_doc_embeddings[doc_indices])

            q_emb = query_embeddings[q_idx]
            reranked = rank.rerank(
                documents_ids=[ranked_ids],
                queries_embeddings=[q_emb],
                documents_embeddings=[query_doc_embeddings],
            )

            # Parse PyLate's output
            for item in reranked[0]:  # list of dicts
                heapq.heappush(
                    result_heaps[qid], (float(item["score"]), str(item["id"]))
                )

            # Keep only top_k
            if len(result_heaps[qid]) > top_k:
                result_heaps[qid] = heapq.nlargest(top_k, result_heaps[qid])

        if self._index_autodelete and self._index_dir is not None:
            try:
                shutil.rmtree(self._index_dir, ignore_errors=True)
            finally:
                self._index_dir = None
                self._index_name = None

        return result_heaps


class MultiVectorModel(AbsEncoder, PylateSearchEncoder):
    task_corpus: CorpusDatasetType | None = None

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        index_dir: str | Path | None = None,
        index_name: str | None = None,
        index_autodelete: bool = True,
        index_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for MultiVector/ColBERT models (via PyLate)."""
        requires_package(self, "pylate", model_name, "pip install mteb[pylate]")
        from pylate.models import ColBERT  # type: ignore[import]

        self.model_name = model_name
        self.model = ColBERT(self.model_name, revision=revision, **kwargs)
        built_in_prompts = getattr(self.model, "prompts", None)
        if built_in_prompts and not model_prompts:
            self.model.prompts = built_in_prompts
        elif model_prompts and built_in_prompts:
            logger.info(f"Model.prompts will be overwritten with {model_prompts}")
            self.model.prompts = self.validate_task_to_prompt_name(model_prompts)

        self.base_index_dir = Path(index_dir) if index_dir else None
        self._index_name = index_name
        self._index_autodelete = index_autodelete
        self.index_kwargs = index_kwargs or {}

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_metadata.name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} prompt_type={prompt_type}"
            )
        logger.debug(f"Encoding {len(inputs)} items.")

        inputs = [text for batch in inputs for text in batch["text"]]

        pred = self.model.encode(
            inputs,
            prompt_name=prompt_name,
            is_query=prompt_type == PromptType.query,
            convert_to_tensor=True,
            **kwargs,
        )

        # encode returns a list of tensors shaped (x, token_dim), pad to uniform length
        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)
        return pred.cpu().numpy()


colbert_v2 = ModelMeta(
    loader=MultiVectorModel,
    name="colbert-ir/colbertv2.0",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-09-21",
    n_parameters=int(110 * 1e6),
    memory_usage_mb=418,
    max_tokens=180,
    embed_dim=None,
    license="mit",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO",
        "mMARCO-NL",
    },
)

jina_colbert_v2 = ModelMeta(
    loader=MultiVectorModel,
    loader_kwargs=dict(
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    ),
    name="jinaai/jina-colbert-v2",
    languages=[
        "ara-Arab",
        "ben-Beng",
        "deu-Latn",
        "spa-Latn",
        "eng-Latn",
        "fas-Arab",
        "fin-Latn",
        "fra-Latn",
        "hin-Deva",
        "ind-Latn",
        "jpn-Jpan",
        "kor-Kore",
        "rus-Cyrl",
        "swa-Latn",
        "tel-Telu",
        "tha-Thai",
        "yor-Latn",
        "zho-Hans",
        "nld-Latn",
        "ita-Latn",
        "por-Latn",
        "vie-Latn",
    ],
    open_weights=True,
    revision="4cf816e5e2b03167b132a3c847a9ecd48ba708e1",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-08-16",
    n_parameters=int(559 * 1e6),
    memory_usage_mb=1067,
    max_tokens=8192,
    embed_dim=None,
    license="cc-by-nc-4.0",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/jinaai/jina-colbert-v2",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO",
        "mMARCO-NL",
        "DuRetrieval",
        "MIRACL",
    },
)


lightonai__gte_moderncolbert_v1 = ModelMeta(
    loader=MultiVectorModel,
    name="lightonai/GTE-ModernColBERT-v1",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="c1647d10d6edc6f70837f42f0a978f2df53f51dd",
    public_training_code="https://gist.github.com/NohTow/3030fe16933d8276dd5b3e9877d89f0f",
    public_training_data="https://huggingface.co/datasets/lightonai/ms-marco-en-bge-gemma",
    release_date="2025-04-30",
    n_parameters=int(149 * 1e6),
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=None,
    license="apache-2.0",
    similarity_fn_name="MaxSim",
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/lightonai/GTE-ModernColBERT-v1",
    use_instructions=False,
    adapted_from="Alibaba-NLP/gte-modernbert-base",
    superseded_by=None,
    training_datasets={
        "MSMARCO",
        "mMARCO-NL",
    },
)
