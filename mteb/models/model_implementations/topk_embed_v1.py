from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        BatchedInput,
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )


class TopkEmbedSearch:
    """Late-interaction search model for the topk-embed-v1 models.

    Multi-vector embeddings only support retrieval-style search, so this implements the
    SearchProtocol, not the Encoder interface. Token vectors are stored in FP16 and scored with
    exhaustive FP32 MaxSim. The models have no image+text fusion: an input with both columns
    (ViDoRe v3.1: page image and OCR markdown) is encoded from its text only.
    """

    mteb_model_meta: ModelMeta | None = None

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        image_token_budget: int = 2048,
        document_chunk_size: int = 256,
        max_score_elements: int = 2**28,
        **kwargs: Any,
    ):
        from sentence_transformers import MultiVectorEncoder

        self.device = device or "cuda"
        self.model = MultiVectorEncoder(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
            config_kwargs={"image_token_budget": image_token_budget},
        )
        applied = self.model[0].config.image_token_budget
        if applied != image_token_budget:
            raise ValueError(
                f"image_token_budget={image_token_budget} was not applied; the model uses {applied}"
            )
        self.document_chunk_size = document_chunk_size
        self.max_score_elements = max_score_elements
        # The query-language subsets of a task share one corpus, so it is encoded once.
        self._corpus_key: str | None = None
        self._corpus_ids: list[str] = []
        self._corpus_vectors: list[torch.Tensor] = []

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        from mteb._create_dataloaders import create_dataloader

        ids = list(corpus["id"])
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        key = f"{task_metadata.name}/{hf_split}/{digest}"
        if key == self._corpus_key:
            return
        loader = create_dataloader(
            corpus,
            task_metadata=task_metadata,
            prompt_type=PromptType.document,
            num_proc=num_proc,
            **encode_kwargs,
        )
        self._corpus_vectors = self._encode(loader, is_query=False)
        self._corpus_ids = ids
        self._corpus_key = key

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
    ) -> RetrievalOutputType:
        import torch

        from mteb._create_dataloaders import create_dataloader

        if self._corpus_key is None:
            raise ValueError("Corpus must be indexed before searching.")
        loader = create_dataloader(
            queries,
            task_metadata=task_metadata,
            prompt_type=PromptType.query,
            num_proc=num_proc,
            **encode_kwargs,
        )
        scores = self._maxsim(self._encode(loader, is_query=True), self._corpus_vectors)
        position = {
            corpus_id: index for index, corpus_id in enumerate(self._corpus_ids)
        }
        results: RetrievalOutputType = {}
        for query_id, row in zip(queries["id"], scores, strict=True):
            if top_ranked is None:
                candidates, candidate_scores = self._corpus_ids, row
            else:
                candidates = top_ranked[query_id]
                rows = torch.tensor([position[c] for c in candidates], dtype=torch.long)
                candidate_scores = row[rows]
            values, indices = candidate_scores.topk(min(top_k, len(candidates)))
            results[query_id] = {
                candidates[index]: value
                for value, index in zip(values.tolist(), indices.tolist(), strict=True)
            }
        return results

    def _encode(
        self, inputs: DataLoader[BatchedInput], *, is_query: bool
    ) -> list[torch.Tensor]:
        import torch

        column = "text" if "text" in inputs.dataset.features else "image"
        # The models are trained on text queries; an image query is encoded like an image document.
        if is_query and column == "text":
            encode = self.model.encode_query
        else:
            encode = self.model.encode_document
        vectors = []
        for batch in inputs:
            items = batch[column]
            if column == "image":
                items = [image.convert("RGB") for image in items]
            output = encode(items, batch_size=len(items), show_progress_bar=False)
            vectors.extend(vector.to(torch.float16).cpu() for vector in output)
        # Vectors stay unpadded: one long page would pad the whole corpus to its length.
        return vectors

    def _maxsim(
        self, queries: list[torch.Tensor], documents: list[torch.Tensor]
    ) -> torch.Tensor:
        import torch

        # MultiVectorEncoder.similarity computes token similarities in the input dtype, so the
        # FP16 vectors are upcast chunk by chunk to score in FP32 without an FP32 copy of the corpus.
        queries = [vector.to(self.device, torch.float32) for vector in queries]
        scores = []
        for start in range(0, len(documents), self.document_chunk_size):
            chunk = [
                vector.to(self.device, torch.float32)
                for vector in documents[start : start + self.document_chunk_size]
            ]
            scores.append(
                self.model.similarity(
                    queries, chunk, chunk_elements=self.max_score_elements
                ).cpu()
            )
        return torch.cat(scores, dim=1)


TOPK_EMBED_V1_LANGUAGES = [
    "eng-Latn",
    "fra-Latn",
    "deu-Latn",
    "spa-Latn",
    "ita-Latn",
    "por-Latn",
]

topk_embed_v1_xsmall = ModelMeta(
    loader=TopkEmbedSearch,
    loader_kwargs={"image_token_budget": 2048},
    name="topk-io/topk-embed-v1-xsmall",
    revision="d09d8a7a8cdd6c287f792b3c4d7b41233d46e66a",
    release_date="2026-09-23",
    model_type=["late-interaction"],
    languages=TOPK_EMBED_V1_LANGUAGES,
    modalities=["image", "text"],
    n_parameters=854_034_496,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=1629,
    max_tokens=8192,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers", "Transformers", "safetensors"],
    reference="https://huggingface.co/topk-io/topk-embed-v1-xsmall",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=None,
    adapted_from="Qwen/Qwen3.5-0.8B",
    superseded_by=None,
    output_dtypes=OutputDType.FLOAT16,
    extra_requirements_groups=["topk-embed"],
)

topk_embed_v1_small = ModelMeta(
    loader=TopkEmbedSearch,
    loader_kwargs={"image_token_budget": 2048},
    name="topk-io/topk-embed-v1-small",
    revision="33b15d544d74f29d04cdb97adeb9fd0e52da5fb7",
    release_date="2026-09-23",
    model_type=["late-interaction"],
    languages=TOPK_EMBED_V1_LANGUAGES,
    modalities=["image", "text"],
    n_parameters=2_217_435_968,
    n_embedding_parameters=508_559_360,
    memory_usage_mb=4229,
    max_tokens=8192,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers", "Transformers", "safetensors"],
    reference="https://huggingface.co/topk-io/topk-embed-v1-small",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=None,
    adapted_from="Qwen/Qwen3.5-2B",
    superseded_by=None,
    output_dtypes=OutputDType.FLOAT16,
    extra_requirements_groups=["topk-embed"],
)
