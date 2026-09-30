from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


class TopkEmbedWrapper(AbsEncoder):
    """Late-interaction wrapper for the topk-embed-v1 models.

    Token vectors are stored in FP16 and scored with exhaustive FP32 MaxSim. The models have no
    image+text fusion: a corpus with both columns (ViDoRe v3.1: page image and OCR markdown) is
    encoded from its text only.
    """

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
        self._corpus_cache: dict[str, list[torch.Tensor]] = {}

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
        import torch

        is_query = prompt_type == PromptType.query
        column = "text" if is_query or "text" in inputs.dataset.features else "image"
        key = None
        if not is_query and "id" in inputs.dataset.column_names:
            digest = hashlib.sha256(
                "\n".join(inputs.dataset["id"]).encode()
            ).hexdigest()
            key = f"{task_metadata.name}/{hf_split}/{column}/{digest}"
            if key in self._corpus_cache:
                return self._corpus_cache[key]
        encode = self.model.encode_query if is_query else self.model.encode_document
        vectors = []
        for batch in inputs:
            items = batch[column]
            if column == "image":
                items = [image.convert("RGB") for image in items]
            output = encode(items, batch_size=len(items), show_progress_bar=False)
            vectors.extend(vector.to(torch.float16).cpu() for vector in output)
        # Vectors stay unpadded: one long page would pad the whole corpus to its length.
        if key is not None:
            self._corpus_cache = {key: vectors}
        return vectors

    def similarity(self, a: Array, b: Array) -> Array:
        import torch

        # MultiVectorEncoder.similarity computes token similarities in the input dtype, so the
        # FP16 vectors are upcast chunk by chunk to score in FP32 without an FP32 copy of the corpus.
        queries = [
            torch.as_tensor(vector).to(self.device, torch.float32) for vector in a
        ]
        scores = []
        for start in range(0, len(b), self.document_chunk_size):
            chunk = [
                torch.as_tensor(vector).to(self.device, torch.float32)
                for vector in b[start : start + self.document_chunk_size]
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
    loader=TopkEmbedWrapper,
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
    loader=TopkEmbedWrapper,
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
