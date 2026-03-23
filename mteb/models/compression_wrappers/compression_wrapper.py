from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import torch

from mteb.types import OutputDType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.model_meta import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class CompressionWrapper:
    """Wraps a model to quantize the embeddings and compute results on the compressed vectors instead."""

    def __init__(
        self,
        model: EncoderProtocol,
        quantization_level: OutputDType = OutputDType.FLOAT8_E4M3FN,
        clipping_margin: tuple[float, float] | None = None,
    ) -> None:
        """Instantiates the wrapper with an embedding model and sets the quantization level.

        Args:
            model: The model to produce quantized embeddings.
            quantization_level: The quantization level to use. Has to be supported by the quantize_embeddings method.
            clipping_margin: Optional lower and upper percentiles to crop embeddings before integer quantization.
        """
        self.model = model
        self._quantization_level = quantization_level
        self.clipping_margin = None
        self.min_embeds = 10_000
        embed_types = model.mteb_model_meta.embedding_types
        model.mteb_model_meta.experiment_kwargs = {
            "precision": quantization_level.value
        }
        model.mteb_model_meta.embedding_types = [quantization_level]
        if clipping_margin is not None:
            assert 0 < clipping_margin[0] < clipping_margin[1] < 1
            self.clipping_margin = torch.tensor(clipping_margin)
        if embed_types and quantization_level in embed_types:
            logger.warning(
                f"The model {model.mteb_model_meta.name} internally supports quantization to "
                f"{quantization_level.value}, which might lead to better results."
            )
            warnings.warn(
                f"The model {model.mteb_model_meta.name} internally supports quantization to "
                f"{quantization_level.value}, which might lead to better results."
            )
        logger.info("Initialized CompressionWrapper.")

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        """Return wrapped model meta data."""
        return self.model.mteb_model_meta

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        """Encodes the given sentences using the encoder, then quantizes the embeddings.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task.
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The name type of prompt. (query or passage)
            batch_size: Batch size
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded and quantized input in an array of the shape (Number of sentences) x (Embedding dimension).
        """
        embeddings = self.model.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings).float()

        logger.info(f"Quantizing embeddings to {self._quantization_level.value}.")
        return self._quantize_embeddings(embeddings)

    def _quantize_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> Array:
        """Compresses embeddings to represent each dimension with lower bit-precision.

        Takes full-precision embeddings as input and quantizes them to the chosen bit range. When quantizing to
        integers, the minimum and maximum values per dimension need to be estimated first. For retrieval tasks, this
        should be done on document embeddings, so the same thresholds can be applied to queries, meaning that (a batch
        of) documents need to be embedded first.

        Args:
            embeddings: The embeddings to quantize.

        Returns:
            The quantized embeddings.
        """
        torch_dtype = self._quantization_level.get_dtype()
        if self._quantization_level in [
            OutputDType.FLOAT8_E4M3FN,
            OutputDType.FLOAT8_E5M2,
            OutputDType.FLOAT8_E8M0FNU,
            OutputDType.FLOAT8_E4M3FNUZ,
            OutputDType.FLOAT8_E5M2FNUZ,
            OutputDType.FLOAT16,
        ]:
            # Cast to float8, then back to float16 using PyTorch as numpy doesn't support float8
            quantized = embeddings.type(torch_dtype).type(torch.float16)
        elif self._quantization_level == OutputDType.BF16:
            # Cast to bf16, then back to float32 using PyTorch as numpy doesn't support bf16
            quantized = embeddings.type(torch_dtype).float()
        elif self._quantization_level in [
            OutputDType.INT8,
            OutputDType.UINT8,
            OutputDType.INT4,
            OutputDType.UINT4,
        ]:
            num_bits = (
                8
                if self._quantization_level in [OutputDType.INT8, OutputDType.UINT8]
                else 4
            )
            if self.clipping_margin is not None:
                cutoffs = torch.quantile(embeddings, self.clipping_margin, dim=0)
                embeddings = torch.clip(embeddings, cutoffs[0], cutoffs[1])
            mins, maxs = self._get_min_max_per_dim(embeddings)
            steps = (maxs - mins) / (2**num_bits - 1)
            subtract = (
                0
                if self._quantization_level in [OutputDType.UINT8, OutputDType.UINT4]
                else int(2**num_bits * 0.5)
            )
            quantized = torch.floor((embeddings - mins) / steps) - subtract
        elif self._quantization_level == OutputDType.BINARY:
            quantized = torch.where(embeddings > 0, 1.0, 0.0)
        else:
            raise ValueError(
                f"Quantization method '{self._quantization_level.value}' is not supported!"
            )
        return quantized

    def _get_min_max_per_dim(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes thresholds for integer quantization.

        Calculates the minimum and maximum values per embedding dimension and returns these. If the prompt type is
        query, then the pre-calculated values are returned instead. If no pre-calculated values exist, an error is
        raised.

        Args:
            embeddings: The embeddings for which minima and maxima should be calculated.

        Returns:
            The minimum and maximum values per embedding dimension.
        """
        if len(embeddings) < self.min_embeds:
            logger.warning(
                f"Estimating quantization parameters on less than {self.min_embeds} embeddings (only "
                f"{len(embeddings)}). Parameters are likely unstable and results might not generalize."
            )
            warnings.warn(
                f"Estimating quantization parameters on less than {self.min_embeds} embeddings (only "
                f"{len(embeddings)}). Parameters are likely unstable and results might not generalize."
            )

        mins, maxs = (
            torch.min(embeddings, dim=0).values,
            torch.max(embeddings, dim=0).values,
        )
        return mins, maxs

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity] for more details."""
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity_pairwise] for more details."""
        return self.model.similarity_pairwise(embeddings1, embeddings2)
