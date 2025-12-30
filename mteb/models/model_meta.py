from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from huggingface_hub import (
    GitCommitInfo,
    ModelCard,
    ModelCardData,
    get_safetensors_metadata,
    hf_hub_download,
    list_repo_commits,
    repo_exists,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
    SafetensorsParsingError,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from transformers import AutoConfig
from typing_extensions import Self

from mteb._helpful_enum import HelpfulStrEnum
from mteb.languages import check_language_code
from mteb.models.models_protocols import EncoderProtocol, MTEBModels
from mteb.types import ISOLanguageScript, Licenses, Modalities, StrDate, StrURL

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from mteb.abstasks import AbsTask


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
]

MODEL_TYPES = Literal["dense", "cross-encoder", "late-interaction"]


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


_SENTENCE_TRANSFORMER_LIB_NAME: FRAMEWORKS = "Sentence Transformers"


class ModelMeta(BaseModel):
    """The model metadata object.

    Attributes:
        loader: The function that loads the model. If None it assumes that the model is not implemented.
        loader_kwargs: The keyword arguments to pass to the loader function.
        name: The name of the model, ideally the name on huggingface. It should be in the format "organization/model_name".
        n_parameters: The number of parameters in the model, e.g. 7_000_000 for a 7M parameter model. Can be None if the number of parameters is not known (e.g. for proprietary models) or
            if the loader returns a SentenceTransformer model from which it can be derived.
        memory_usage_mb: The memory usage of the model in MB. Can be None if the memory usage is not known (e.g. for proprietary models). To calculate it use the `calculate_memory_usage_mb` method.
        max_tokens: The maximum number of tokens the model can handle. Can be None if the maximum number of tokens is not known (e.g. for proprietary
            models).
        embed_dim: The dimension of the embeddings produced by the model. Currently all models are assumed to produce fixed-size embeddings.
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
    memory_usage_mb: float | None
    max_tokens: float | None
    embed_dim: int | None
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

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_is_cross_encoder(cls, data: Any) -> Any:
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
        if v is None or v in ("bm25s", "Human"):
            return v
        if "/" not in v:
            raise ValueError(
                "Model name must be in the format 'organization/model_name'"
            )
        return v

    def load_model(self, **kwargs: Any) -> MTEBModels:
        """Loads the model using the specified loader function."""
        if self.loader is None:
            raise NotImplementedError(
                "No model implementation is available for this model."
            )
        if self.name is None:
            raise ValueError("name is not set for ModelMeta. Cannot load model.")

        # Allow overwrites
        _kwargs = self.loader_kwargs.copy()
        _kwargs.update(kwargs)

        model: MTEBModels = self.loader(self.name, revision=self.revision, **_kwargs)
        model.mteb_model_meta = self  # type: ignore[misc]
        return model

    def model_name_as_path(self) -> str:
        """Returns the model name in a format that can be used as a file path.

        Replaces "/" with "__" and spaces with "_".
        """
        if self.name is None:
            raise ValueError("Model name is not set")
        return self.name.replace("/", "__").replace(" ", "_")

    @classmethod
    def _from_hub(
        cls,
        model_name: str | None,
        revision: str | None = None,
        compute_metadata: bool = True,
    ) -> Self:
        """Generates a ModelMeta from a HuggingFace model name.

        Args:
            model_name: The HuggingFace model name.
            revision: Revision of the model
            compute_metadata: Add metadata based on model card

        Returns:
            The generated ModelMeta.
        """
        from mteb.models import sentence_transformers_loader

        loader = sentence_transformers_loader
        frameworks: list[FRAMEWORKS] = ["PyTorch"]
        model_license = None
        reference = None
        n_parameters = None
        memory_usage_mb = None
        release_date = None
        embedding_dim = None
        max_tokens = None

        if model_name and compute_metadata and _repo_exists(model_name):
            reference = "https://huggingface.co/" + model_name
            card = ModelCard.load(model_name)
            card_data: ModelCardData = card.data
            try:
                model_config = AutoConfig.from_pretrained(model_name)
            except Exception as e:
                # some models can't load AutoConfig (e.g. `average_word_embeddings_levy_dependency`)
                model_config = None
                logger.warning(f"Can't get configuration for {model_name}. Error: {e}")

            if card_data.library_name == _SENTENCE_TRANSFORMER_LIB_NAME or (
                card_data.tags and _SENTENCE_TRANSFORMER_LIB_NAME in card_data.tags
            ):
                frameworks.append(_SENTENCE_TRANSFORMER_LIB_NAME)
            else:
                msg = "Model library not recognized, defaulting to Sentence Transformers loader."
                logger.warning(msg)
                warnings.warn(msg)

            if revision is None:
                revisions = _get_repo_commits(model_name, "model")
                revision = revisions[0].commit_id if revisions else None

            release_date = cls.fetch_release_date(model_name)
            model_license = card_data.license
            n_parameters = cls._calculate_num_parameters_from_hub(model_name)
            memory_usage_mb = cls._calculate_memory_usage_mb(model_name, n_parameters)
            if model_config and hasattr(model_config, "hidden_size"):
                embedding_dim = model_config.hidden_size
            if model_config and hasattr(model_config, "max_position_embeddings"):
                max_tokens = model_config.max_position_embeddings

        return cls(
            loader=loader,
            name=model_name or "no_model_name/available",
            revision=revision or "no_revision_available",
            reference=reference,
            release_date=release_date,
            languages=None,
            license=model_license,
            framework=frameworks,
            training_datasets=None,
            similarity_fn_name=None,
            n_parameters=n_parameters,
            memory_usage_mb=memory_usage_mb,
            max_tokens=max_tokens,
            embed_dim=embedding_dim,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            modalities=[],
        )

    @classmethod
    def from_sentence_transformer_model(
        cls,
        model: SentenceTransformer,
        revision: str | None = None,
        compute_metadata: bool = True,
    ) -> Self:
        """Generates a ModelMeta from a SentenceTransformer model.

        Args:
            model: SentenceTransformer model.
            revision: Revision of the model
            compute_metadata: Add metadata based on model card

        Returns:
            The generated ModelMeta.
        """
        name: str | None = (
            model.model_card_data.model_name
            if model.model_card_data.model_name
            else model.model_card_data.base_model
        )
        meta = cls._from_hub(name, revision, compute_metadata)
        if _SENTENCE_TRANSFORMER_LIB_NAME not in meta.framework:
            meta.framework.append("Sentence Transformers")
        meta.revision = model.model_card_data.base_model_revision or meta.revision
        meta.max_tokens = model.max_seq_length
        meta.embed_dim = model.get_sentence_embedding_dimension()
        meta.similarity_fn_name = ScoringFunction.from_str(model.similarity_fn_name)
        meta.modalities = ["text"]
        return meta

    @classmethod
    def from_hub(
        cls,
        model: str,
        revision: str | None = None,
        compute_metadata: bool = True,
    ) -> Self:
        """Generates a ModelMeta for model from HuggingFace hub.

        Args:
            model: Name of the model from HuggingFace hub. For example, `intfloat/multilingual-e5-large`
            revision: Revision of the model
            compute_metadata: Add metadata based on model card

        Returns:
            The generated ModelMeta.
        """
        meta = cls._from_hub(model, revision, compute_metadata)
        if _SENTENCE_TRANSFORMER_LIB_NAME not in meta.framework:
            meta.framework.append("Sentence Transformers")
        meta.modalities = ["text"]

        if model and compute_metadata and _repo_exists(model):
            # have max_seq_length field
            sbert_config = _get_json_from_hub(
                model, "sentence_bert_config.json", "model", revision=revision
            )
            if sbert_config:
                meta.max_tokens = (
                    sbert_config.get("max_seq_length", None) or meta.max_tokens
                )
            # have model type, similarity function fields
            config_sbert = _get_json_from_hub(
                model, "config_sentence_transformers.json", "model", revision=revision
            )
            if (
                config_sbert is not None
                and config_sbert.get("similarity_fn_name") is not None
            ):
                meta.similarity_fn_name = ScoringFunction.from_str(
                    config_sbert["similarity_fn_name"]
                )
            else:
                meta.similarity_fn_name = ScoringFunction.COSINE
        return meta

    @classmethod
    def from_cross_encoder(
        cls,
        model: CrossEncoder,
        revision: str | None = None,
        compute_metadata: bool = True,
    ) -> Self:
        """Generates a ModelMeta from a CrossEncoder.

        Args:
            model: The CrossEncoder model
            revision: Revision of the model
            compute_metadata: Add metadata based on model card

        Returns:
            The generated ModelMeta
        """
        from mteb.models import CrossEncoderWrapper

        meta = cls._from_hub(model.model.name_or_path, revision, compute_metadata)
        if _SENTENCE_TRANSFORMER_LIB_NAME not in meta.framework:
            meta.framework.append("Sentence Transformers")
        meta.revision = model.config._commit_hash or meta.revision
        meta.loader = CrossEncoderWrapper
        meta.embed_dim = None
        meta.modalities = ["text"]
        meta.model_type = ["cross-encoder"]
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
            tasks = cast(Sequence["AbsTask"], tasks)
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
            tasks = cast(Sequence["AbsTask"], tasks)
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
    def _calculate_memory_usage_mb(
        model_name: str, n_parameters: int | None
    ) -> int | None:
        MB = 1024**2  # noqa: N806
        try:
            safetensors_metadata = get_safetensors_metadata(model_name)
            if len(safetensors_metadata.parameter_count) >= 0:
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

    def calculate_memory_usage_mb(self) -> int | None:
        """Calculates the memory usage of the model in MB.

        Returns:
            The memory usage of the model in MB, or None if it cannot be determined.
        """
        if "API" in self.framework or self.name is None:
            return None

        return self._calculate_memory_usage_mb(self.name, self.n_parameters)

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

    def to_python(self) -> str:
        """Returns a string representation of the model."""
        return _pydantic_instance_to_code(self)


def _pydantic_instance_to_code(
    model: BaseModel,
    indent: int = 4,
    *,
    only_set_fields: bool = False,
) -> str:
    """Convert a Pydantic model instance into valid Python constructor code.

    If only_set_fields=True, only fields explicitly provided at model construction
    time are printed (i.e., excludes fields that came only from defaults).

    Arguments:
        model: The Pydantic model to convert.
        indent: The indentation to use.
        only_set_fields: If True, only fields explicitly provided at model construction time
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


def _get_repo_commits(repo_id: str, repo_type: str) -> list[GitCommitInfo] | None:
    try:
        return list_repo_commits(repo_id=repo_id, repo_type=repo_type)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        logger.warning(f"Can't get commits of {repo_id}: {e}")
        return None


def _get_json_from_hub(
    repo_id: str, file_name: str, repo_type: str, revision: str | None = None
) -> dict[str, Any] | None:
    path = _get_file_on_hub(repo_id, file_name, repo_type, revision)
    if path is None:
        return None

    with Path(path).open() as f:
        js = json.load(f)
    return js


def _get_file_on_hub(
    repo_id: str, file_name: str, repo_type: str, revision: str | None = None
) -> str | None:
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=file_name, repo_type=repo_type, revision=revision
        )
    except (GatedRepoError, RepositoryNotFoundError, EntryNotFoundError) as e:
        logger.warning(f"Can't get file {file_name} of {repo_id}: {e}")
        return None


def _repo_exists(repo_id: str, repo_type: str | None = None) -> bool:
    """Checks if a repository exists on HuggingFace Hub.

    Repo exists will raise HFValidationError for invalid local paths

    Args:
        repo_id: The repository ID.
        repo_type: The type of repository (e.g., "model", "dataset", "space").
    """
    try:
        return repo_exists(repo_id=repo_id, repo_type=repo_type)
    except HFValidationError as e:
        logger.warning(f"Can't check existence of {repo_id}: {e}")
        return False
