from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from packaging.version import Version

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    _concatenate_ragged_batches,
    _encode_batches,
    _multi_vector_similarity,
    _select_encode_function,
)
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from PIL.Image import Image
    from sentence_transformers.multi_vector_encoder.modules import (
        HierarchicalTokenPooling,
    )
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs


def _format_passage_batches(inputs: DataLoader[BatchedInput]) -> Iterator[BatchedInput]:
    for batch in inputs:
        if "title" in batch and "body" in batch:
            columns = cast("dict[str, list[str]]", batch)
            texts = [
                f"title: {title or ''} | text: {body}"
                for title, body in zip(columns["title"], columns["body"], strict=True)
            ]
            yield cast("BatchedInput", {**columns, "text": texts})
        else:
            yield batch


def _limit_image(image: Image, max_side: int | None) -> Image:
    if max_side is None or max(image.size) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


class FridaV2(AbsEncoder):
    """Encode either head of the FRIDA v2 Sentence Transformers checkpoint."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        *,
        vector_type: Literal["single_vector", "multi_vector"] = "multi_vector",
        embed_dim: int | None = None,
        token_pooling: Literal["HierarchicalTokenPooling"] | None = None,
        token_pooling_args: dict[str, Any] | None = None,
        max_image_size: int | None = 2000,
        **kwargs: Any,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if vector_type not in {"single_vector", "multi_vector"}:
            raise ValueError(f"Unknown vector_type {vector_type!r}.")

        dense = vector_type == "single_vector"
        self._supported_dims = [512, 1024, 2048] if dense else [32, 64, 128]
        if embed_dim is not None and embed_dim not in self._supported_dims:
            raise ValueError(
                f"{vector_type} supports embed_dim in {self._supported_dims}."
            )
        if dense and token_pooling is not None:
            raise ValueError("Token pooling requires the multi_vector head.")

        self.token_pooling = _load_token_pooling(token_pooling)
        self._token_pooling_config = (
            dict(token_pooling) if token_pooling is not None else None
        )
        self.embed_dim = embed_dim
        self.max_image_size = max_image_size
        self.vector_type = vector_type
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            **kwargs,
        )
        self._mteb_model_meta: ModelMeta | None

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta | None) -> None:
        if meta is None:
            self._mteb_model_meta = None
            return
        dense = self.vector_type == "single_vector"
        experiment_kwargs = dict(meta.experiment_kwargs or {})
        if dense:
            experiment_kwargs["vector_type"] = self.vector_type
        else:
            experiment_kwargs.pop("vector_type", None)
        if self._token_pooling_config is not None:
            experiment_kwargs["token_pooling"] = self._token_pooling_config
        self._mteb_model_meta = meta.model_copy(
            update={
                "embed_dim": self.embed_dim or self._supported_dims,
                "modalities": ["text", "image"],
                "max_tokens": self.model.max_seq_length,
                "model_type": ["dense" if dense else "late-interaction"],
                "similarity_fn_name": (
                    ScoringFunction.COSINE if dense else ScoringFunction.MAX_SIM
                ),
                "experiment_kwargs": experiment_kwargs or None,
            },
            deep=True,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Encode with the checkpoint's query or document prompt."""
        import torch

        encode = _select_encode_function(self.model, prompt_type)
        dense = self.vector_type == "single_vector"
        if not dense and kwargs.get("precision") not in {None, "float32"}:
            raise ValueError("Multi-vector embeddings do not support quantization.")

        def encode_batch(
            rows: list[dict[str, Any]], **batch_kwargs: Any
        ) -> Array | list[torch.Tensor]:
            rows = [
                {**row, "image": _limit_image(row["image"], self.max_image_size)}
                if "image" in row
                else row
                for row in rows
            ]
            if dense:
                return cast(
                    "Array", encode(rows, truncate_dim=self.embed_dim, **batch_kwargs)
                )

            features = encode(rows, output_value=None, **batch_kwargs)
            embeddings = [
                item["multivector_embeddings"][item["attention_mask"].bool()].to(
                    device="cpu", dtype=torch.float32
                )
                for item in features
            ]
            if self.embed_dim is not None:
                embeddings = [
                    torch.nn.functional.normalize(tokens[:, : self.embed_dim], dim=-1)
                    for tokens in embeddings
                ]
            if self.token_pooling is not None:
                embeddings = self.token_pooling.pool(
                    embeddings,
                )
            return embeddings

        is_multimodal = "image" in inputs.dataset.features  # type: ignore[attr-defined]
        batches = (
            _format_passage_batches(inputs)
            if prompt_type == PromptType.document and not is_multimodal
            else inputs
        )
        return _encode_batches(
            batches,
            is_multimodal=is_multimodal,
            encode_function=encode_batch,
            prompt=None,
            modalities=self._mteb_model_meta.modalities,
            concatenate_batches=None if dense else _concatenate_ragged_batches,
            encode_text_per_batch=not dense,
            **kwargs,
        )

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Score all pairs using cosine or the device-blocked MaxSim search."""
        if self.vector_type == "single_vector":
            return super().similarity(embeddings1, embeddings2)
        from sentence_transformers.util import maxsim

        return _multi_vector_similarity(
            embeddings1,
            embeddings2,
            device=self.model.device,
            score_function=maxsim,
        )

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Score corresponding pairs on the model's device for MaxSim."""
        if self.vector_type == "single_vector":
            return super().similarity_pairwise(embeddings1, embeddings2)
        from sentence_transformers.util import maxsim_pairwise

        return maxsim_pairwise(embeddings1, embeddings2, device=self.model.device)


# delete after merge https://github.com/embeddings-benchmark/mteb/pull/5991
def _load_token_pooling(
    config: dict[str, Any] | None,
) -> HierarchicalTokenPooling | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError("token_pooling must be a configuration dictionary.")

    import sentence_transformers

    if Version(sentence_transformers.__version__).release < (6, 1, 0):
        raise ImportError("Token pooling requires sentence-transformers >= 6.1.0.")

    from sentence_transformers.multi_vector_encoder.modules import (
        HierarchicalTokenPooling,
    )

    return HierarchicalTokenPooling(**config)


frida_v2 = ModelMeta(
    loader=FridaV2,
    name="ai-forever/FRIDA-Embeddings-V2-2B",
    revision="43bbb92fff0c90e09df6351a8fd81ac5d1d34109",
    release_date="2026-11-28",
    languages=["rus-Cyrl", "eng-Latn"],
    n_parameters=2213241664,
    n_embedding_parameters=508559360,
    memory_usage_mb=4221,
    max_tokens=4096,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/ai-forever/FRIDA-Embeddings-V2-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=set(),
    adapted_from="Qwen/Qwen3.5-2B",
    modalities=["text", "image"],
    model_type=["dense"],
    contacts=["Samoed"],
    output_dtypes=OutputDType.FLOAT32,
    extra_requirements_groups=None,
)
