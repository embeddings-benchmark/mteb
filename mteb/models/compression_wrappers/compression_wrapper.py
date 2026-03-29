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
    """Wraps a model to quantize the embeddings and compute results on the compressed vectors instead.

    Examples:
        >>> import mteb
        >>> from mteb.models import CompressionWrapper
        >>> from mteb.types import OutputDType
        >>> model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> compression_model = CompressionWrapper(model, OutputDType.INT8)
        >>> task = mteb.get_task("NanoArguAnaRetrieval")
        >>> mteb.evaluate(compression_model, task)
    """

    def __init__(
        self,
        model: EncoderProtocol,
        output_dtype: OutputDType = OutputDType.FLOAT8_E4M3FN,
        clipping_margin: tuple[float, float] | None = None,
    ) -> None:
        """Instantiates the wrapper with an embedding model and sets the quantization level.

        Args:
            model: The model to produce quantized embeddings.
            output_dtype: The output data type to compress to. Has to be supported by the quantize_embeddings method.
            clipping_margin: Optional lower and upper percentiles to crop embeddings before integer quantization.
        """
        self.model = model
        self._quantization_level = output_dtype
        self.clipping_margin = None
        self.min_embeds = 10_000
        meta = model.mteb_model_meta
        embed_types = meta.output_dtypes
        exp_kwargs = dict(meta.experiment_kwargs) if meta.experiment_kwargs else {}
        exp_kwargs["output_dtypes"] = output_dtype.value

        if clipping_margin is not None:
            if not 0 < clipping_margin[0] < clipping_margin[1] < 1:
                raise ValueError(
                    f"Clipping margin must be between 0 and 1 with lower bound {clipping_margin[0]} < "
                    f"upper bound {clipping_margin[1]}, but got {clipping_margin}."
                )
            self.clipping_margin = torch.tensor(clipping_margin)
            exp_kwargs["clipping_margin"] = list(clipping_margin)

        model.mteb_model_meta = meta.model_copy(
            update={
                "output_dtypes": [output_dtype],
                "experiment_kwargs": exp_kwargs,
            }
        )  # type: ignore[misc]
        if embed_types and output_dtype in embed_types:
            msg = (
                f"The model {model.mteb_model_meta.name} natively supports quantization to {output_dtype.value} and "
                f"can be configured to return a compressed embedding vector without using the wrapper. Please note "
                f"that performance on compressed embedding might be better when using compressed embeddings returned "
                f"directly by the model."
            )
            logger.warning(msg)
            warnings.warn(msg)
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
        """Encodes the given inputs using the encoder, then quantizes the embeddings.

        Generates embeddings for the given inputs, then compresses them based on the specified output dtype. While
        embeddings returned by this function are compressed to the value range determined by the output type, it returns
        32- or 16-bit floats to avoid issues with potential downstream calculations and array conversions.

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
        integers, the minimum and maximum values per dimension need to be estimated first.

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

        Calculates the minimum and maximum values per embedding dimension and returns these. The values are used to
        estimate the bin boundaries used to map floating points to discrete integer values. If the number of passed
        embeddings is small, a warning is raised that the calculated values might be unstable.

        Args:
            embeddings: The embeddings for which minima and maxima should be calculated.

        Returns:
            The minimum and maximum values per embedding dimension.
        """
        if len(embeddings) < self.min_embeds:
            msg = (
                f"Estimating quantization parameters on less than {self.min_embeds} embeddings (only "
                f"{len(embeddings)}). Parameters are likely unstable and results might not generalize."
            )
            logger.warning(msg)
            warnings.warn(msg)

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
