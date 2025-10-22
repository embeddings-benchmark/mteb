import logging
from collections.abc import Callable, Sequence
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, cast

from huggingface_hub import get_safetensors_metadata
from huggingface_hub.errors import (
    GatedRepoError,
    NotASafetensorsRepoError,
    SafetensorsParsingError,
)
from pydantic import BaseModel, ConfigDict, field_validator

from mteb.languages import check_language_code
from mteb.types import ISOLanguageScript, Licenses, Modalities, StrDate, StrURL

from .models_protocols import EncoderProtocol, MTEBModels

if TYPE_CHECKING:
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


class ScoringFunction(str, Enum):
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
        n_parameters: The number of parameters in the model, e.g. 7_000_000 for a 7M parameter model. Can be None if the number of parameters is not known (e.g. for proprietary models) or
            if the loader returns a SentenceTransformer model from which it can be derived.
        memory_usage_mb: The memory usage of the model in MB. Can be None if the memory usage is not known (e.g. for proprietary models). To calculate it use the `calculate_memory_usage_mb` method.
        max_tokens: The maximum number of tokens the model can handle. Can be None if the maximum number of tokens is not known (e.g. for proprietary
            models).
        embed_dim: The dimension of the embeddings produced by the model. Currently all models are assumed to produce fixed-size embeddings.
        revision: The revision number of the model. If None, it is assumed that the metadata (including the loader) is valid for all revisions of the model.
        release_date: The date the model's revision was released.
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
        is_cross_encoder: Whether the model can act as a cross-encoder or not.
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
    is_cross_encoder: bool | None = None
    citation: str | None = None
    contacts: list[str] | None = None

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

        model: EncoderProtocol = self.loader(
            self.name, revision=self.revision, **_kwargs
        )
        model.mteb_model_meta = self  # type: ignore
        return model

    def model_name_as_path(self) -> str:
        """Returns the model name in a format that can be used as a file path.

        Replaces "/" with "__" and spaces with "_".
        """
        if self.name is None:
            raise ValueError("Model name is not set")
        return self.name.replace("/", "__").replace(" ", "_")

    def is_zero_shot_on(
        self, tasks: Sequence["AbsTask"] | Sequence[str]
    ) -> bool | None:
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
                logger.warning(f"Could not get source model: {e} in MTEB")

        return_dataset = training_datasets.copy()
        visited = set()

        for dataset in training_datasets:
            similar_tasks = _collect_similar_tasks(dataset, visited)
            return_dataset |= similar_tasks

        return return_dataset

    def zero_shot_percentage(
        self, tasks: Sequence["AbsTask"] | Sequence[str]
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

    def calculate_memory_usage_mb(self) -> int | None:
        """Calculates the memory usage (in FP32) of the model in MB.

        Returns:
            The memory usage of the model in MB, or None if it cannot be determined.
        """
        if "API" in self.framework:
            return None

        MB = 1024**2  # noqa: N806
        try:
            safetensors_metadata = get_safetensors_metadata(self.name)  # type: ignore
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

        except (NotASafetensorsRepoError, SafetensorsParsingError, GatedRepoError):
            pass
        if self.n_parameters is None:
            return None
        # Model memory in bytes. For FP32 each parameter is 4 bytes.
        model_memory_bytes = self.n_parameters * 4

        # Convert to MB
        model_memory_mb = model_memory_bytes / MB
        return round(model_memory_mb)


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
