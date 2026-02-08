import logging
from typing import Any

import torch
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
        quantization_level: str,
    ) -> None:
        """Init

        Args:
            model: The model to produce quantized embeddings.
            quantization_level: The quantization level to use. Has to be supported by the quantize_embeddings method.
        """
        self.model = model
        self._quantization_level = quantization_level
        self.mins, self.maxs = None, None
        self.min_embeds = 10_000
        self.query_embeds = None
        self.hf_subset = None
        self.task_name = None
        embed_types = None
        if model.mteb_model_meta:
            embed_types = model.mteb_model_meta.embedding_types
            model.mteb_model_meta.name = (
                f"{model.mteb_model_meta.name} (output_dtype={quantization_level})"
            )
        if embed_types and quantization_level in embed_types:
            logger.warning(
                f"The model {model.mteb_model_meta.name} internally supports quantization to {quantization_level}. "
                f"Setting output file name to {model.mteb_model_meta.name} (output_dtype={quantization_level}*) to "
                f"avoid conflict."
            )
            model.mteb_model_meta.name = (
                f"{model.mteb_model_meta.name} (output_dtype={quantization_level}*)"
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
            The encoded and quantized input in a numpy array of the shape (Number of sentences) x (Embedding dimension).
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

        if self.hf_subset != hf_subset or task_metadata.name != self.task_name:
            self.hf_subset = hf_subset
            self.task_name = task_metadata.name
            self._reset_boundaries()

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        if prompt_type == PromptType.query and task_metadata.category in [
            "t2i",
            "i2t",
            "it2i",
            "i2it",
        ]:
            # With multimodal tasks, always quantize text and image embeddings separately
            logger.info(f"Quantizing query embeddings to {self._quantization_level}")
            return self._quantize_embeddings(embeddings, PromptType.document)
        elif prompt_type == PromptType.query and self._quantization_level in [
            "int8",
            "int4",
        ]:
            # Otherwise, compute thresholds for int8/int4 quantization on documents first, then apply them on queries
            logger.info("Query embeddings will be quantized on similarity calculation.")
            self.query_embeds = embeddings
            return embeddings
        else:
            logger.info(f"Quantizing embeddings to {self._quantization_level}")
            return self._quantize_embeddings(embeddings, prompt_type)

    def _quantize_embeddings(
        self,
        embeddings: torch.tensor,
        prompt_type: PromptType | None = None,
    ) -> Array:
        """Compresses embeddings to represent each dimension with lower bit-precision.

        Takes full-precision embeddings as input and quantizes them to the chosen bit range. When quantizing to
        integers, the minimum and maximum values per dimension need to be estimated first. For retrieval tasks, this
        should be done on document embeddings, so the same thresholds can be applied to queries, meaning that (a batch
        of) documents need to be embedded first.

        Args:
            embeddings: The embeddings to quantize.
            prompt_type: The name type of prompt. (query or passage)

        Returns:
            The quantized embeddings.
        """
        quantiles = torch.tensor([0.025, 0.975])
        if self._quantization_level == "float8":
            # Cast to float8, then back to float16 using PyTorch as numpy doesn't support float8
            quantized = embeddings.type(torch.float8_e4m3fn).type(torch.float16)
        elif self._quantization_level == "int8" or self._quantization_level == "int4":
            num_bits = 8 if self._quantization_level == "int8" else 4
            cutoffs = torch.quantile(embeddings, quantiles, dim=0)
            quantized = torch.clip(embeddings, cutoffs[0], cutoffs[1])
            mins, maxs = self._get_min_max_per_dim(embeddings, prompt_type)
            steps = (maxs - mins) / (2**num_bits - 1)
            quantized = torch.floor((quantized - mins) / steps) - int(2**num_bits * 0.5)
        elif self._quantization_level == "binary":
            quantized = torch.where(embeddings > 0, 1.0, 0.0)
        else:
            raise ValueError(
                f"Quantization method {self._quantization_level} is not supported!"
            )
        return quantized.numpy()

    def _get_min_max_per_dim(
        self,
        embeddings: torch.tensor,
        prompt_type: PromptType | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes thresholds for integer quantization.

        Calculates the minimum and maximum values per embedding dimension and returns these. If the prompt type is
        query, then the pre-calculated values are returned instead. If no pre-calculated values exist, an error is
        raised.

        Args:
            embeddings: The embeddings for which minima and maxima should be calculated.
            prompt_type: The name type of prompt. (query or passage)

        Returns:
            The minimum and maximum values per embedding dimension.
        """
        if prompt_type == PromptType.query:
            if self.mins is not None and self.maxs is not None:
                return self.mins, self.maxs
            else:
                raise ValueError(
                    "Quantizing query embeddings requires pre-computed thresholds! Make sure to calculate quantized "
                    "document embeddings first."
                )
        # Use values calculated on first set of text for tasks like bitext mining or classification
        elif self.mins is not None and self.maxs is not None:
            return self.mins, self.maxs
        else:
            if len(embeddings) < self.min_embeds:
                logger.warning(
                    f"Estimating quantization parameters on less than {self.min_embeds} embeddings (only "
                    f"{len(embeddings)}). Parameters are likely unstable and results might not generalize."
                )
            mins, maxs = (
                torch.min(embeddings, dim=0).values,
                torch.max(embeddings, dim=0).values,
            )
            self.mins = mins
            self.maxs = maxs
            return mins, maxs

    def _reset_boundaries(self) -> None:
        """Resets the minima and maxima for evaluation on multiple datasets or batches."""
        self.mins = None
        self.maxs = None

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity] for more details."""
        if self.query_embeds is not None:
            if not isinstance(embeddings1, torch.Tensor):
                embeddings1 = torch.tensor(embeddings1)
            if not isinstance(embeddings2, torch.Tensor):
                embeddings2 = torch.tensor(embeddings2)
            logger.info("Quantizing query embeddings.")
            if torch.equal(embeddings1, self.query_embeds):
                embeddings1 = self._quantize_embeddings(embeddings1, PromptType.query)
            elif torch.equal(embeddings2, self.query_embeds):
                embeddings2 = self._quantize_embeddings(embeddings2, PromptType.query)
        self._reset_boundaries()
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity_pairwise] for more details."""
        self._reset_boundaries()
        return self.model.similarity_pairwise(embeddings1, embeddings2)
