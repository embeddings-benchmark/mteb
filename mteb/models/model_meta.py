from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Callable, Mapping
from dataclasses import field
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from huggingface_hub import (
    ModelCard,
    get_safetensors_metadata,
    model_info,
)
from huggingface_hub.errors import (
    GatedRepoError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
    SafetensorsParsingError,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoConfig

from mteb._helpful_enum import HelpfulStrEnum
from mteb._hf_integration.hf_hub_utils import (
    _get_json_from_hub,
    _get_repo_commits,
    _repo_exists,
)
from mteb.languages import check_language_code
from mteb.models.models_protocols import MTEBModels
from mteb.types import ISOLanguageScript, Licenses, Modalities, StrDate, StrURL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from huggingface_hub import (
        ModelCardData,
    )
    from typing_extensions import Self

    from mteb.abstasks import AbsTask
    from mteb.cache import ResultCache
    from mteb.models.models_protocols import EncoderProtocol


logger = logging.getLogger(__name__)

FRAMEWORKS = Literal[
    "Sentence Transformers",
    "PyTorch",
    "GritLM",
    "LLM2Vec",
    "TensorFlow",
    "API",
    "Tevatron",
    "NumPy",
    "PyLate",
    "ColBERT",
    "ColPali",
    "GGUF",
    "safetensors",
    "ONNX",
    "Transformers",
]

MODEL_TYPES = Literal["dense", "cross-encoder", "late-interaction", "sparse", "router"]


class ScoringFunction(HelpfulStrEnum):
    """The scoring function used by the models."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    MAX_SIM = "MaxSim"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CUSTOM = "custom"


def _get_loader_name(
    loader: Callable[..., EncoderProtocol] | None,
) -> str | None:
    if loader is None:
        return None
    if hasattr(loader, "func"):  # partial class wrapper
        return loader.func.__name__
    return loader.__name__


class ModelMeta(BaseModel):
    """The model metadata object.

    Attributes:
        loader: The function that loads the model. If None it assumes that the model is not implemented.
        loader_kwargs: The keyword arguments to pass to the loader function.
        name: The name of the model, ideally the name on huggingface. It should be in the format "organization/model_name".
        n_parameters: The total number of parameters in the model, e.g. `7_000_000` for a 7M parameter model. Can be none in case the number of parameters is unknown.
        n_embedding_parameters: The number of parameters used for the embedding layer. Can be None if the number of embedding parameters is not known (e.g. for proprietary models).
        n_active_parameters_override: The number of active parameters used bu model. Should be used **only** for Mixture of Experts models.
        memory_usage_mb: The memory usage of the model in MB. Can be None if the memory usage is not known (e.g. for proprietary models). To calculate it use the `calculate_memory_usage_mb` method.
        max_tokens: The maximum number of tokens the model can handle. Can be None if the maximum number of tokens is not known (e.g. for proprietary
            models).
        embed_dim: The dimension of the embeddings produced by the model. Currently all models are assumed to produce fixed-size embeddings.
          If annotated as list this will be treated as a range of possible embedding dimensions (Matryoshka).
        revision: The revision number of the model. If None, it is assumed that the metadata (including the loader) is valid for all revisions of the model.
        release_date: The date the model's revision was released. If None, then release date will be added based on 1st commit in hf repository of model.
        license: The license under which the model is released. Required if open_weights is True.
        open_weights: Whether the model is open source or proprietary.
        public_training_code: A link to the publicly available training code. If None, it is assumed that the training code is not publicly available.
        public_training_data: A link to the publicly available training data. If None, it is assumed that the training data is not publicly available.
        similarity_fn_name: The distance metric used by the model.
        framework: The framework the model is implemented in, can be a list of frameworks e.g. `["Sentence Transformers", "PyTorch"]`.
        reference: A URL to the model's page on huggingface or another source.
        languages: The languages the model is intended to be specified as a 3-letter language code followed by a script code e.g., "eng-Latn" for English
            in the Latin script.
        use_instructions: Whether the model uses instructions E.g. for prompt-based models. This also includes models that require a specific format for
            input, such as "query: {document}" or "passage: {document}".
        citation: The citation for the model. This is a bibtex string.
        training_datasets: A dictionary of datasets that the model was trained on. Names should be names as their appear in `mteb` for example
            {"ArguAna"} if the model is trained on the ArguAna test set. This field is used to determine if a model generalizes zero-shot to
            a benchmark as well as mark dataset contaminations.
        adapted_from: Name of the model from which this model is adapted. For quantizations, fine-tunes, long doc extensions, etc.
        superseded_by: Name of the model that supersedes this model, e.g., nvidia/NV-Embed-v2 supersedes v1.
        model_type: A list of strings representing the type of model.
        modalities: A list of strings representing the modalities the model supports. Default is ["text"].
        contacts: The people to contact in case of a problem in the model, preferably a GitHub handle.
        experiment_kwargs: A dictionary of parameters used in the experiment that are not covered by other fields. This is used to create experiment names for ablation studies and similar experiments.
    """

    model_config = ConfigDict(extra="forbid")

    # loaders
    loader: Callable[..., MTEBModels] | None
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str | None
    revision: str | None
    release_date: StrDate | None
    languages: list[ISOLanguageScript] | None
    n_parameters: int | None
    n_active_parameters_override: int | None = None
    n_embedding_parameters: int | None = None
    memory_usage_mb: float | None
    max_tokens: float | None
    embed_dim: int | list[int] | None
    license: Licenses | StrURL | None
    open_weights: bool | None
    public_training_code: str | None
    public_training_data: str | bool | None
    framework: list[FRAMEWORKS]
    reference: StrURL | None = None
    similarity_fn_name: ScoringFunction | None
    use_instructions: bool | None
    training_datasets: set[str] | None
    adapted_from: str | None = None
    superseded_by: str | None = None
    modalities: list[Modalities] = ["text"]
    model_type: list[MODEL_TYPES] = ["dense"]
    citation: str | None = None
    contacts: list[str] | None = None
    experiment_kwargs: Mapping[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_is_cross_encoder(cls, data: Any) -> Any:
        """Handle legacy is_cross_encoder field by converting it to model_type.

        This validator handles backward compatibility for the deprecated is_cross_encoder field.
        If is_cross_encoder=True is provided, it adds "cross_encoder" to model_type.
        """
        if isinstance(data, dict) and "is_cross_encoder" in data:
            is_cross_encoder_value = data.pop("is_cross_encoder")

            if is_cross_encoder_value is not None:
                warnings.warn(
                    "is_cross_encoder is deprecated and will be removed in a future version. "
                    "Use model_type=['cross-encoder'] instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

                model_type = data.get("model_type", ["dense"])

                if is_cross_encoder_value:
                    if "cross-encoder" not in model_type:
                        data["model_type"] = ["cross-encoder"]
                else:
                    if "cross-encoder" in model_type:
                        model_type = [t for t in model_type if t != "cross-encoder"]
                        data["model_type"] = model_type if model_type else ["dense"]

        return data

    @property
    def is_cross_encoder(self) -> bool:
        """Returns True if the model is a cross-encoder.

        Derived from model_type field. A model is considered a cross-encoder if "cross-encoder" is in its model_type list.
        """
        return "cross-encoder" in self.model_type

    @property
    def n_active_parameters(self):
        """Number of active parameters. Assumed to be `n_parameters - n_embedding_parameters`. Can be overwritten using `n_active_parameters_override` e.g. for MoE models."""
        if self.n_active_parameters_override is not None:
            return self.n_active_parameters_override

        if self.n_parameters is not None and self.n_embedding_parameters is not None:
            return self.n_parameters - self.n_embedding_parameters
        return None

    @field_validator("similarity_fn_name", mode="before")
    @classmethod
    def _validate_similarity_fn_name(cls, value: str) -> ScoringFunction | None:
        """Converts the similarity function name to the corresponding enum value.

        Sentence_transformers uses Literal['cosine', 'dot', 'euclidean', 'manhattan'],
        and pylate uses Literal['MaxSim']

        Args:
            value: The similarity function name as a string.

        Returns:
            The corresponding ScoringFunction enum value.
        """
        if type(value) is ScoringFunction or value is None:
            return value
        mapping = {
            "cosine": ScoringFunction.COSINE,
            "dot": ScoringFunction.DOT_PRODUCT,
            "MaxSim": ScoringFunction.MAX_SIM,
        }
        if value in mapping:
            return mapping[value]
        raise ValueError(f"Invalid similarity function name: {value}")

    def to_dict(self):
        """Returns a dictionary representation of the model metadata."""
        dict_repr = self.model_dump()
        loader = dict_repr.pop("loader", None)
        dict_repr["training_datasets"] = (
            list(dict_repr["training_datasets"])
            if isinstance(dict_repr["training_datasets"], set)
            else dict_repr["training_datasets"]
        )
        dict_repr["loader"] = _get_loader_name(loader)
        dict_repr["is_cross_encoder"] = self.is_cross_encoder
        return dict_repr

    @field_validator("languages")
    @classmethod
    def _languages_are_valid(
        cls, languages: list[ISOLanguageScript] | None
    ) -> list[ISOLanguageScript] | None:
        if languages is None:
            return None

        for code in languages:
            check_language_code(code)
        return languages

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if "/" not in v:
            raise ValueError(
                "Model name must be in the format 'organization/model_name'"
            )
        return v

    def load_model(
        self,
        device: str | None = None,
        *,
        embed_dim: int | None = None,
        **kwargs: Any,
    ) -> MTEBModels:
        """Loads the model using the specified loader function."""
        if self.loader is None:
            raise NotImplementedError(
                "No model implementation is available for this model."
            )
        if self.name is None:
            raise ValueError("name is not set for ModelMeta. Cannot load model.")

        if embed_dim is not None:
            if (
                self.embed_dim is not None
                and isinstance(self.embed_dim, int)
                and self.embed_dim != embed_dim
            ):
                raise ValueError(
                    f"Requested embedding dimension {embed_dim} does not match the model's embedding dimension {self.embed_dim}."
                    "Model does not support loading with a different embedding dimension."
                )
            elif isinstance(self.embed_dim, list) and embed_dim not in self.embed_dim:
                raise ValueError(
                    f"Requested embedding dimension {embed_dim} is not in the model's supported embedding dimensions {self.embed_dim}."
                )
            self.embed_dim = embed_dim
            if self.experiment_kwargs is None:
                self.experiment_kwargs = {"embed_dim": embed_dim}
            else:
                self.experiment_kwargs["embed_dim"] = embed_dim

        if self.experiment_kwargs is None:
            self.experiment_kwargs = kwargs if len(kwargs) > 0 else None
        elif len(kwargs) > 0 and self.experiment_kwargs is not None:
            kwargs |= self.experiment_kwargs
            self.experiment_kwargs = kwargs

        # Allow overwrites
        _kwargs = self.loader_kwargs.copy()
        _kwargs.update(kwargs)
        if device is not None:
            _kwargs["device"] = device

        model: MTEBModels = self.loader(
            self.name,
            revision=self.revision,
            embed_dim=embed_dim,
            **_kwargs,
        )
        model.mteb_model_meta = self  # type: ignore[misc]
        return model

    def model_name_as_path(self) -> str:
        """Returns the model name in a format that can be used as a file path.

        Replaces "/" with "__" and spaces with "_".
        """
        if self.name is None:
            raise ValueError("Model name is not set")
        return self.name.replace("/", "__").replace(" ", "_")

    @property
    def experiment_name(self) -> str | None:
        """Create a filesystem-safe string representation of the experiment parameters.

        Uses deterministic serialization and hashing to ensure stable, bounded output.

        Examples:
            >>> import mteb
            >>> model = mteb.get_model("mteb/baseline-random-encoder", param1="test")
            >>>
            >>> print(model.mteb_model_meta.experiment_name)
            >>> # param1_test
        """
        return _serialize_experiment_kwargs_to_name(
            experiment_kwargs=self.experiment_kwargs
        )

    @property
    def model_name_with_experiment(self) -> str | None:
        """Combines the model name with the experiment parameters for a more descriptive name."""
        if self.name is None:
            return None
        experiment_str = _serialize_experiment_kwargs_to_name(
            experiment_kwargs=self.experiment_kwargs,
            value_field_separator="=",
            kwargs_separator=", ",
        )
        return f"{self.name} ({experiment_str})" if experiment_str else self.name

    @classmethod
    def _detect_cross_encoder_or_dense(
        cls,
        model_name: str,
        revision: str | None,
        config: dict[str, Any] | None,
        sentence_transformers_loader: Callable[..., MTEBModels],
        cross_encoder_loader: Callable[..., MTEBModels],
    ) -> tuple[Callable[..., MTEBModels] | None, MODEL_TYPES]:
        """Detect if model is CrossEncoder or default to dense."""
        if not config:
            logger.warning(
                f"Could not load config.json for {model_name}. "
                "Defaulting to SentenceTransformer loader."
            )
            return sentence_transformers_loader, "dense"

        architectures = config.get("architectures", [])

        is_cross_encoder = any(
            arch.endswith("ForSequenceClassification") for arch in architectures
        )
        if is_cross_encoder:
            return cross_encoder_loader, "cross-encoder"

        if cls._is_causal_lm_reranker(architectures, config, model_name):
            return cross_encoder_loader, "cross-encoder"

        logger.info(
            f"Model {model_name} does not have modules.json or recognized architecture. "
            "Defaulting to SentenceTransformer loader."
        )
        return sentence_transformers_loader, "dense"

    @staticmethod
    def _is_causal_lm_reranker(
        architectures: list[str], config: dict[str, Any], model_name: str
    ) -> bool:
        """Check if model is a CausalLM-style reranker."""
        is_causal_lm = any(arch.endswith("ForCausalLM") for arch in architectures)

        if not is_causal_lm:
            return False

        num_labels = config.get("num_labels", 0)
        model_name_lower = model_name.lower()

        return (
            num_labels > 0
            or "rerank" in model_name_lower
            or "cross-encoder" in model_name_lower
        )

    @classmethod
    def _detect_model_type_and_loader(
        cls,
        model_name: str,
        revision: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[Callable[..., MTEBModels] | None, MODEL_TYPES]:
        """Detect the model type and appropriate loader based on HuggingFace Hub configuration files.

        This follows the Sentence Transformers architecture detection logic:
        1. Check for modules.json - If present, model is a SentenceTransformer (dense encoder)
        2. If no modules.json, check config.json for architecture:
            - ForSequenceClassification → CrossEncoder
            - CausalLM with reranking indicators → CrossEncoder
        3. Default to dense (SentenceTransformer) if no clear indicators are found

        Detection for CausalLM-style rerankers:
        - Model has ForCausalLM architecture AND
        - Has num_labels > 0 in config, OR
        - Model name contains "rerank" or "cross-encoder"

        Args:
            model_name: The HuggingFace model name
            revision: The model revision
            config: The loaded config.json from the HuggingFace model repository. If not provided, it will be fetched from the hub.


        Returns:
            A tuple of (loader_function, model_type) where:
            - loader_function: A callable that returns MTEBModels, or None if model doesn't exist
            - model_type: One of "dense", "cross-encoder", or "late-interaction"
        """
        from mteb.models import CrossEncoderWrapper, sentence_transformers_loader

        try:
            modules_config = _get_json_from_hub(
                model_name, "modules.json", "model", revision=revision
            )

            if (
                modules_config
            ):  # SentenceTransformer/SparseEncoder (Not support for now)
                return sentence_transformers_loader, "dense"
            else:
                return cls._detect_cross_encoder_or_dense(
                    model_name,
                    revision,
                    config,
                    sentence_transformers_loader,
                    cross_encoder_loader=CrossEncoderWrapper,
                )

        except Exception as e:
            logger.warning(
                f"Error detecting model type for {model_name}: {e}. "
                "Defaulting to SentenceTransformer loader."
            )

        return sentence_transformers_loader, "dense"

    @classmethod
    def create_empty(cls, overwrites: dict[str, Any] | None = None) -> Self:
        """Creates an empty ModelMeta with all fields set to None or empty."""
        empty_model = cls(
            loader=None,
            name=None,
            revision=None,
            release_date=None,
            languages=None,
            n_parameters=None,
            n_embedding_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            reference=None,
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
            citation=None,
            contacts=None,
        )
        if overwrites:
            empty_model = empty_model.model_copy(update=overwrites)

        if empty_model.name is None:
            empty_model.name = "no_model_name/available"
        if empty_model.revision is None:
            empty_model.revision = "no_revision_available"

        return empty_model

    def merge(self, overwrite: Self) -> Self:
        """Merges another this ModelMeta with another ModelMeta.

        Args:
            overwrite: The ModelMeta to merge into this one. Non-None fields in this ModelMeta will overwrite the corresponding fields in this
                ModelMeta. the `framework` and `model_type` fields with combined.

        Returns:
            A new ModelMeta with the merged fields.
        """
        merged_data = self.model_dump()
        overwrite_data = overwrite.model_dump()

        for key, value in overwrite_data.items():
            if (
                key == "name"
                and value == "no_model_name/available"
                and self.name != "no_model_name/available"
            ):
                continue  # skip overwriting name if overwrite has no name available
            if (
                key == "revision"
                and value == "no_revision_available"
                and self.revision != "no_revision_available"
            ):
                continue  # skip overwriting revision if overwrite has no revision available
            if key in ["framework", "model_type"]:
                # Combine lists and remove duplicates
                merged_list = set(merged_data.get(key, [])) | set(value or [])
                merged_data[key] = list(merged_list)
            if value is not None:
                merged_data[key] = value

        return self.model_copy(update=merged_data)

    @classmethod
    def _from_sentence_transformer_model(cls, model: SentenceTransformer) -> Self:
        """Generates a ModelMeta from only a SentenceTransformer model, without fetching any additional metadata from HuggingFace Hub."""
        from mteb.models import sentence_transformers_loader

        name: str | None = (
            model.model_card_data.model_name
            if model.model_card_data.model_name
            else model.model_card_data.base_model
        )
        n_embedding_parameters = (
            cls._get_n_embedding_parameters_from_sentence_transformers(model)
        )
        return cls.create_empty(
            overwrites=dict(
                name=name,
                revision=model.model_card_data.base_model_revision,
                loader=sentence_transformers_loader,
                max_tokens=model.max_seq_length,
                embed_dim=model.get_sentence_embedding_dimension(),
                similarity_fn_name=ScoringFunction.from_str(model.similarity_fn_name),
                framework=["Sentence Transformers", "PyTorch"],
                n_embedding_parameters=n_embedding_parameters,
            )
        )

    @staticmethod
    def _get_n_embedding_parameters_from_sentence_transformers(
        model: SentenceTransformer | CrossEncoder,
    ) -> int | None:
        """Calculates the number of embedding parameters in a SentenceTransformer model

        This is based on the heuristic: `vocab_size * embedding_dim` where vocab_size and embedding_dim are extracted from the model's first
        Transformer module.
        """
        logger.info(
            "Calculating number of embedding parameters for SentenceTransformer model."
        )

        emb = None
        if isinstance(model, CrossEncoder) and hasattr(
            model.model, "get_input_embeddings"
        ):
            emb = model.model.get_input_embeddings()
            return int(np.prod(emb.weight.shape))
        elif isinstance(model, SentenceTransformer):
            vocab = None
            try:
                vocab = len(model.tokenizer.vocab)
            except Exception as e:
                msg = f"Could not determine vocab size for model {model.model_card_data.model_name} and therefore cannot calculate number of embedding parameters. \nError: \n{e}"
                logger.warning(msg)
            embedding_dimensions = model.get_sentence_embedding_dimension()
            if embedding_dimensions is not None and vocab is not None:
                return vocab * embedding_dimensions

        logger.warning(
            f"Model does not have a recognized architecture for calculating embedding parameters (model={model.model_card_data.model_name})."
        )
        return None

    @classmethod
    def _from_cross_encoder_model(cls, model: CrossEncoder) -> Self:
        """Generates a ModelMeta from only a CrossEncoder model, without fetching any additional metadata from HuggingFace Hub."""
        from mteb.models import CrossEncoderWrapper

        return cls.create_empty(
            overwrites=dict(
                loader=CrossEncoderWrapper,
                name=model.model.name_or_path,
                revision=model.config._commit_hash,
                framework=["Sentence Transformers", "PyTorch"],
                model_type=["cross-encoder"],
                n_embedding_parameters=cls._get_n_embedding_parameters_from_sentence_transformers(
                    model
                ),
            )
        )

    @classmethod
    def _from_hub(
        cls,
        model_name: str,
        revision: str | None = None,
    ) -> Self:
        """Generates a ModelMeta from a HuggingFace model name.

        Args:
            model_name: The HuggingFace model name.
            revision: Revision of the model
            fill_missing: Fill missing attributes from the metadata including number of parameters and memory usage.

        Returns:
            The generated ModelMeta.
        """
        loader: Callable[..., MTEBModels] | None
        model_type: MODEL_TYPES

        reference = "https://huggingface.co/" + model_name

        if not _repo_exists(model_name):
            warnings.warn(
                f"Could not find model {model_name} on HuggingFace Hub repository ({reference}). Metadata will be limited."
            )
            return cls.create_empty(
                overwrites=dict(
                    name=model_name,
                    revision=revision,
                )
            )
        config = _get_json_from_hub(
            model_name, "config.json", "model", revision=revision
        )
        loader, model_type = cls._detect_model_type_and_loader(
            model_name, revision, config=config
        )
        card = ModelCard.load(model_name)
        card_data = card.data
        card_data = cast("ModelCardData", card_data)
        try:
            model_config = AutoConfig.from_pretrained(model_name)
        except Exception as e:
            # some models can't load AutoConfig (e.g. `average_word_embeddings_levy_dependency`)
            model_config = None
            logger.warning(
                f"Can't get model configuration for {model_name}. Error: {e}"
            )

        frameworks = cls._get_frameworks_from_hf_tags(model_name) if model_name else []

        if revision is None:
            revisions = _get_repo_commits(model_name, "model")
            revision = revisions[0].commit_id if revisions else None

        model_license = card_data.license if card_data.license != "other" else None
        n_parameters = cls._calculate_num_parameters_from_hub(model_name)
        n_embedding_parameters = cls._estimate_embedding_parameters_from_hub(
            model_name, revision=revision, config=config
        )
        memory_usage_mb = cls._calculate_memory_usage_mb(
            model_name, n_parameters, fetch_from_hf=True
        )

        embedding_dim = getattr(model_config, "hidden_size", None)
        max_tokens = getattr(model_config, "max_position_embeddings", None)

        sbert_config = _get_json_from_hub(
            model_name, "sentence_bert_config.json", "model", revision=revision
        )
        if sbert_config:
            if max_tokens is None:
                max_tokens = sbert_config.get("max_seq_length", None)
        # have model type, similarity function fields
        config_sbert = _get_json_from_hub(
            model_name, "config_sentence_transformers.json", "model", revision=revision
        )
        similarity_fn_name = (
            ScoringFunction.from_str(config_sbert["similarity_fn_name"])
            if config_sbert is not None
            and config_sbert.get("similarity_fn_name") is not None
            else ScoringFunction.COSINE
        )

        return cls.create_empty(
            overwrites=dict(
                loader=loader,
                name=model_name,
                model_type=[model_type],
                revision=revision,
                reference=reference,
                release_date=cls.fetch_release_date(model_name),
                license=model_license,
                framework=frameworks,
                n_parameters=n_parameters,
                n_embedding_parameters=n_embedding_parameters,
                memory_usage_mb=memory_usage_mb,
                max_tokens=max_tokens,
                embed_dim=embedding_dim,
                similarity_fn_name=similarity_fn_name,
            )
        )

    @classmethod
    def from_sentence_transformer_model(
        cls,
        model: SentenceTransformer,
        revision: str | None = None,
        fill_missing: bool = False,
        compute_metadata: bool | None = None,
        fetch_from_hf: bool = False,
    ) -> Self:
        """Generates a ModelMeta from a SentenceTransformer model.

        Args:
            model: SentenceTransformer model.
            revision: Revision of the model
            fill_missing: Fill missing attributes from the metadata including number of parameters and memory usage.
            compute_metadata: Deprecated. Use fill_missing instead.
            fetch_from_hf: Whether to fetch additional metadata from HuggingFace Hub based on the model name. If False, only metadata that can be
                extracted from the SentenceTransformer model will be used.

        Returns:
            The generated ModelMeta.
        """
        if compute_metadata is not None:
            warnings.warn(
                "The compute_metadata parameter is deprecated and will be removed in a future version. "
                f"Use fetch_from_hf instead. Setting `fetch_from_hf={compute_metadata}`.",
                DeprecationWarning,
                stacklevel=2,
            )
            fetch_from_hf = compute_metadata

        if fill_missing is not None:
            warnings.warn(
                "The fill_missing parameter is deprecated and will be removed in a future version. "
                f"Use fetch_from_hf instead. Setting `fetch_from_hf={fill_missing}`.",
                DeprecationWarning,
                stacklevel=2,
            )
            fetch_from_hf = fill_missing

        meta = cls._from_sentence_transformer_model(model)
        if fetch_from_hf:
            if meta.name is None:
                logger.warning(
                    "Model name is not set in metadata extracted from SentenceTransformer model. Cannot fetch additional metadata from HuggingFace Hub."
                )
            else:
                name = cast("str", meta.name)
                meta_hub = cls._from_hub(name, revision)
                # prioritize metadata from the model card but fill missing fields from the hub
                meta = meta_hub.merge(meta)

        return meta

    @classmethod
    def from_hub(
        cls,
        model: str,
        revision: str | None = None,
        fill_missing: bool | None = None,
        compute_metadata: bool | None = None,
    ) -> Self:
        """Generates a ModelMeta for model from HuggingFace hub.

        Args:
            model: Name of the model from HuggingFace hub. For example, `intfloat/multilingual-e5-large`
            revision: Revision of the model
            fill_missing: Deprecated. The fill missing did not add any functionality for this function, but was added for compatibility with
                'from_sentence_transformer_model' and `from_cross_encoder`. It will be removed in a future version.
            compute_metadata: Deprecated. Was superseded by fill_missing.

        Returns:
            The generated ModelMeta.
        """
        if compute_metadata is not None:
            warnings.warn(
                "The compute_metadata parameter is deprecated and will be removed in a future version. It will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        if fill_missing is not None:
            warnings.warn(
                "The fill_missing parameter is deprecated and will be removed in a future version. It will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        meta = cls._from_hub(
            model,
            revision,
        )

        return meta

    @classmethod
    def from_cross_encoder(
        cls,
        model: CrossEncoder,
        revision: str | None = None,
        fill_missing: bool | None = None,
        compute_metadata: bool | None = None,
        fetch_from_hf: bool = False,
    ) -> Self:
        """Generates a ModelMeta from a CrossEncoder.

        Args:
            model: The CrossEncoder model
            revision: Revision of the model
            fill_missing: Fill missing attributes from the metadata including number of parameters and memory usage.
            compute_metadata: Deprecated. Use fill_missing instead.
            fetch_from_hf: Whether to fetch additional metadata from HuggingFace Hub based on the model name. If False, only metadata that can be
                extracted from the CrossEncoder model will be used.

        Returns:
            The generated ModelMeta
        """
        if compute_metadata is not None:
            warnings.warn(
                "The compute_metadata parameter is deprecated and will be removed in a future version. "
                f"Use fetch_from_hf instead. Setting `fetch_from_hf={compute_metadata}`.",
                DeprecationWarning,
                stacklevel=2,
            )
            fetch_from_hf = compute_metadata
        if fill_missing is not None:
            warnings.warn(
                "The fill_missing parameter is deprecated and will be removed in a future version. "
                f"Use fill_missing instead. Setting `fill_missing={fill_missing}`.",
                DeprecationWarning,
                stacklevel=2,
            )
            fetch_from_hf = fill_missing

        meta = cls._from_cross_encoder_model(model)
        if fetch_from_hf:
            name = cast("str", meta.name)
            meta_hub = cls._from_hub(name, revision)
            # prioritize metadata from the model card but fill missing fields from the hub
            meta = meta_hub.merge(meta)

        return meta

    def is_zero_shot_on(self, tasks: Sequence[AbsTask] | Sequence[str]) -> bool | None:
        """Indicates whether the given model can be considered zero-shot or not on the given tasks.

        Returns:
             None if no training data is specified on the model.
        """
        # If no tasks were specified, we're obviously zero-shot
        if not tasks:
            return True
        training_datasets = self.get_training_datasets()
        # If no tasks were specified, we're obviously zero-shot
        if training_datasets is None:
            return None

        if isinstance(tasks[0], str):
            benchmark_datasets = set(tasks)
        else:
            tasks = cast("Sequence[AbsTask]", tasks)
            benchmark_datasets = set()
            for task in tasks:
                benchmark_datasets.add(task.metadata.name)
        intersection = training_datasets & benchmark_datasets
        return len(intersection) == 0

    def get_training_datasets(self) -> set[str] | None:
        """Returns all training datasets of the model including similar tasks."""
        import mteb

        if self.training_datasets is None:
            return None

        training_datasets = self.training_datasets.copy()
        if self.adapted_from is not None:
            try:
                adapted_from_model = mteb.get_model_meta(
                    self.adapted_from, fetch_from_hf=False
                )
                adapted_training_datasets = adapted_from_model.get_training_datasets()
                if adapted_training_datasets is not None:
                    training_datasets |= adapted_training_datasets
            except (ValueError, KeyError) as e:
                msg = f"Could not get source model: {e} in MTEB"
                logger.warning(msg)
                warnings.warn(msg)

        return_dataset = training_datasets.copy()
        visited: set[str] = set()

        for dataset in training_datasets:
            similar_tasks = _collect_similar_tasks(dataset, visited)
            return_dataset |= similar_tasks

        return return_dataset

    def zero_shot_percentage(
        self, tasks: Sequence[AbsTask] | Sequence[str]
    ) -> int | None:
        """Indicates how out-of-domain the selected tasks are for the given model.

        Args:
            tasks: A sequence of tasks or dataset names to evaluate against.

        Returns:
            An integer percentage (0-100) indicating how out-of-domain the tasks are for the model.
            Returns None if no training data is specified on the model or if no tasks are provided.
        """
        training_datasets = self.get_training_datasets()
        if (training_datasets is None) or (not tasks):
            return None
        if isinstance(tasks[0], str):
            benchmark_datasets = set(tasks)
        else:
            tasks = cast("Sequence[AbsTask]", tasks)
            benchmark_datasets = {task.metadata.name for task in tasks}
        overlap = training_datasets & benchmark_datasets
        perc_overlap = 100 * (len(overlap) / len(benchmark_datasets))
        return int(100 - perc_overlap)

    @staticmethod
    def _calculate_num_parameters_from_hub(model_name: str | None = None) -> int | None:
        if not model_name:
            return None
        try:
            safetensors_metadata = get_safetensors_metadata(model_name)
            if len(safetensors_metadata.parameter_count) >= 0:
                return sum(safetensors_metadata.parameter_count.values())
        except (
            NotASafetensorsRepoError,
            SafetensorsParsingError,
            GatedRepoError,
            RepositoryNotFoundError,
        ) as e:
            logger.warning(
                f"Can't calculate number of parameters for {model_name}. Got error {e}"
            )
        return None

    def calculate_num_parameters_from_hub(self) -> int | None:
        """Calculates the number of parameters in the model.

        Returns:
            Number of parameters in the model.
        """
        return self._calculate_num_parameters_from_hub(self.name)

    @staticmethod
    def _estimate_embedding_parameters_from_hub(
        model_name: str | None = None,
        revision: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> int | None:
        """Calculate the number of embedding parameters from the model config (vocab_size * hidden_size).  Note that this is an heuristic that works for many models, but might be incorrect.

        Returns:
            Number of embedding parameters in the model.
        """
        if not model_name:
            return None

        if not config:
            logger.warning(
                f"Could not calculate embedding parameters for {model_name} as config.json could not be loaded"
            )
            return None

        vocab_size = config.get("vocab_size")
        if vocab_size is None and "text_config" in config:
            vocab_size = config["text_config"].get("vocab_size")

        if vocab_size is None:
            logger.warning(
                f"Could not calculate embedding parameters for {model_name} as vocab_size is missing from config"
            )
            return None

        hidden_size = config.get("hidden_size") or config.get("hidden_dim")
        if hidden_size is None and "text_config" in config:
            hidden_size = config["text_config"].get("hidden_size") or config[
                "text_config"
            ].get("hidden_dim")

        if hidden_size is None:
            logger.warning(
                f"Could not calculate embedding parameters for {model_name} as hidden_size/hidden_dim is missing from config"
            )
            return None
        return vocab_size * hidden_size

    @staticmethod
    def _calculate_memory_usage_mb(
        model_name: str,
        n_parameters: int | None,
        *,
        fetch_from_hf: bool = False,
    ) -> int | None:
        MB = 1024**2  # noqa: N806

        if fetch_from_hf:
            try:
                safetensors_metadata = get_safetensors_metadata(model_name)
                if safetensors_metadata.parameter_count:
                    dtype_size_map = {
                        "F64": 8,  # 64-bit float
                        "F32": 4,  # 32-bit float (FP32)
                        "F16": 2,  # 16-bit float (FP16)
                        "BF16": 2,  # BFloat16
                        "I64": 8,  # 64-bit integer
                        "I32": 4,  # 32-bit integer
                        "I16": 2,  # 16-bit integer
                        "I8": 1,  # 8-bit integer
                        "U8": 1,  # Unsigned 8-bit integer
                        "BOOL": 1,  # Boolean (assuming 1 byte per value)
                    }
                    total_memory_bytes = sum(
                        parameters * dtype_size_map.get(dtype, 4)
                        for dtype, parameters in safetensors_metadata.parameter_count.items()
                    )
                    return round(total_memory_bytes / MB)  # Convert to MB
            except (
                NotASafetensorsRepoError,
                SafetensorsParsingError,
                GatedRepoError,
                RepositoryNotFoundError,
            ) as e:
                logger.warning(
                    f"Can't calculate memory usage for {model_name}. Got error {e}"
                )

        if n_parameters is None:
            return None
        # Model memory in bytes. For FP32 each parameter is 4 bytes.
        model_memory_bytes = n_parameters * 4

        # Convert to MB
        model_memory_mb = model_memory_bytes / MB
        return round(model_memory_mb)

    def calculate_memory_usage_mb(self, fetch_from_hf: bool = False) -> int | None:
        """Calculates the memory usage of the model in MB.

        Args:
            fetch_from_hf: If True, fetch safetensors metadata from HuggingFace Hub
                to get precise dtype-aware memory usage. If False (default), estimate
                from n_parameters assuming FP32 (4 bytes per parameter).

        Returns:
            The memory usage of the model in MB, or None if it cannot be determined.
        """
        if "API" in self.framework or self.name is None:
            return None

        return self._calculate_memory_usage_mb(
            self.name, self.n_parameters, fetch_from_hf=fetch_from_hf
        )

    @staticmethod
    def fetch_release_date(model_name: str) -> StrDate | None:
        """Fetches the release date from HuggingFace Hub based on the first commit.

        Returns:
            The release date in YYYY-MM-DD format, or None if it cannot be determined.
        """
        commits = _get_repo_commits(repo_id=model_name, repo_type="model")
        if commits:
            initial_commit = commits[-1]
            release_date = initial_commit.created_at.strftime("%Y-%m-%d")
            return release_date
        return None

    @staticmethod
    def _get_frameworks_from_hf_tags(model_name: str) -> list[FRAMEWORKS]:
        """Extract frameworks supported by the model from HuggingFace model tags.

        Args:
            model_name: HuggingFace model name

        Returns:
            List of framework names found in tags. Defaults to empty list if no frameworks found.
        """
        try:
            info = model_info(model_name)
            if not info.tags:
                return []
        except Exception as e:
            logger.warning(
                f"Failed to fetch frameworks from HuggingFace tags for {model_name}: {e}"
            )
            return []

        # Mapping from HuggingFace tags to MTEB framework names
        tag_to_framework: dict[str, FRAMEWORKS] = {
            "sentence-transformers": "Sentence Transformers",
            "transformers": "Transformers",
            "onnx": "ONNX",
            "safetensors": "safetensors",
            "gguf": "GGUF",
        }

        # Assume PyTorch support by default
        # TODO: could be detected from repo as well: https://github.com/embeddings-benchmark/mteb/issues/4104
        frameworks: list[FRAMEWORKS] = ["PyTorch"]

        for framework_tag in tag_to_framework.keys():
            if framework_tag in info.tags:
                frameworks.append(tag_to_framework[framework_tag])

        return frameworks

    def to_python(self) -> str:
        """Returns a string representation of the model."""
        return _pydantic_instance_to_code(self, exclude_fields=["experiment_kwargs"])

    def push_eval_results(
        self,
        user: str | None = None,
        *,
        tasks: Sequence[AbsTask] | Sequence[str] | None = None,
        cache: ResultCache | None = None,
        create_pr: bool = False,
    ) -> None:
        """Pushes the evaluation results of the model to the HuggingFace Hub.

        Args:
            user: The user or organization of results source.
            tasks: The tasks to push results for. If None, results for all tasks will be pushed.
            cache: The ResultCache containing the evaluation results to push.
            create_pr: Whether to create a pull request for the model card update if the model card already exists on the HuggingFace Hub. If False, the model card will be updated directly without a pull request.
        """
        from mteb.cache import ResultCache

        if cache is None:
            cache = ResultCache()

        benchmark_result = cache.load_results(
            models=[self],
            tasks=tasks,
        )
        model_result = benchmark_result.model_results[0]
        model_result.push_model_results(
            user=user,
            create_pr=create_pr,
        )


def _pydantic_instance_to_code(
    model: BaseModel,
    indent: int = 4,
    *,
    only_set_fields: bool = False,
    exclude_fields: Sequence[str] | None = None,
) -> str:
    """Convert a Pydantic model instance into valid Python constructor code.

    If only_set_fields=True, only fields explicitly provided at model construction
    time are printed (i.e., excludes fields that came only from defaults).

    Arguments:
        model: The Pydantic model to convert.
        indent: The indentation to use.
        only_set_fields: If True, only fields explicitly provided at model construction time
        exclude_fields: Fields to exclude from the output, regardless of whether they were set or not.
    """
    cls_name = model.__class__.__name__
    pad = " " * indent
    lines: list[str] = [f"{cls_name}("]

    model_fields = list(type(model).model_fields.keys())

    if only_set_fields:
        field_names = [n for n in model_fields if n in model.model_fields_set]
    else:
        field_names = model_fields

    for field_name in field_names:
        if exclude_fields and field_name in exclude_fields:
            continue
        value = getattr(model, field_name)
        value_code = _value_to_code(value, indent)
        lines.append(f"{pad}{field_name}={value_code},")

    lines.append(")")
    return "\n".join(lines)


def _value_to_code(value: Any, indent: int) -> str:
    """Convert a Python value into valid Python source code."""
    if isinstance(value, BaseModel):
        return _pydantic_instance_to_code(value, indent, only_set_fields=True)

    if callable(value):
        if isinstance(value, partial):
            return value.func.__name__
        return value.__name__

    if isinstance(value, Enum):
        return f"{value.__class__.__name__}.{value.name}"

    if isinstance(value, str):
        return repr(value)

    if isinstance(value, list):
        if not value:
            return "[]"
        inner = ", ".join(_value_to_code(v, indent) for v in value)
        return f"[{inner}]"

    if isinstance(value, set):
        if not value:
            return "set()"
        inner = ", ".join(_value_to_code(v, indent) for v in sorted(value))
        return f"{{{inner}}}"

    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = ", ".join(
            f"{_value_to_code(k, indent)}: {_value_to_code(v, indent)}"
            for k, v in value.items()
        )
        return f"{{{inner}}}"

    return repr(value)


def _collect_similar_tasks(dataset: str, visited: set[str]) -> set[str]:
    """Recursively collect all similar tasks for a given dataset.

    Args:
        dataset: The dataset for which to find similar tasks.
        visited: A set to keep track of visited datasets to avoid cycles.

    Returns:
        A set of similar tasks.
    """
    from mteb.get_tasks import _SIMILAR_TASKS

    if dataset in visited:
        return set()

    visited.add(dataset)
    similar = set()

    # Check if dataset is a key in SIMILAR_TASKS
    if dataset in _SIMILAR_TASKS:
        for similar_task in _SIMILAR_TASKS[dataset]:
            similar.add(similar_task)
            similar.update(_collect_similar_tasks(similar_task, visited))

    # Check if dataset appears as a value in SIMILAR_TASKS
    for parent, children in _SIMILAR_TASKS.items():
        if dataset in children:
            similar.add(parent)
            similar.update(_collect_similar_tasks(parent, visited))

    return similar


def _serialize_experiment_kwargs_to_name(
    experiment_kwargs: Mapping[str, Any] | None,
    value_field_separator: str = "_",
    kwargs_separator: str = "__",
) -> str | None:
    if experiment_kwargs is None or len(experiment_kwargs) == 0:
        return None

    invalid_chars = set('<>:"|?*\\/\0')

    def _serialize_value(value: Any) -> str:
        """Convert value to deterministic string representation."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            value = str(value)
            for invalid_char in invalid_chars:
                value = value.replace(invalid_char, "_")
            return value
        if isinstance(value, (list, tuple)):
            return f"[{','.join(_serialize_value(v) for v in value)}]"
        if isinstance(value, dict):
            items = sorted(value.items())
            return f"{{{','.join(f'{k}:{_serialize_value(v)}' for k, v in items)}}}"
        if isinstance(value, Enum):
            return f"{value.__class__.__name__}.{value.name}"

        # Handle common scientific types
        if hasattr(value, "__module__") and value.__module__ == "numpy":
            # numpy arrays and scalars
            return f"np_{hashlib.sha256(np.asarray(value).tobytes()).hexdigest()[:8]}"

        # Handle pydantic models and dataclasses
        if isinstance(value, BaseModel):
            # Use model_dump for deterministic JSON representation
            json_str = json.dumps(value.model_dump(), sort_keys=True)
            digest = hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:8]
            return f"{value.__class__.__name__}_{digest}"

        raise ValueError(
            f"experiment_kwargs contains non-serializable type {type(value).__name__}. "
            f"Only JSON-serializable types (str, int, float, bool, list, dict, None), "
            f"Enums, numpy arrays, and Pydantic models are supported."
        )

    params_str = kwargs_separator.join(
        f"{key}{value_field_separator}{_serialize_value(value)}"
        for key, value in sorted(experiment_kwargs.items())
    )

    # If too long or contains invalid chars, use hash
    max_length = 200

    if len(params_str) > max_length or any(c in invalid_chars for c in params_str):
        param_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        return f"exp_{param_hash}"

    return params_str
