from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from huggingface_hub import get_safetensors_metadata
from huggingface_hub.errors import (
    GatedRepoError,
    NotASafetensorsRepoError,
    SafetensorsParsingError,
)
from pydantic import BaseModel, ConfigDict, field_validator

from mteb.abstasks.AbsTask import AbsTask
from mteb.encoder_interface import Encoder

from .custom_validators import LICENSES, MODALITIES, STR_DATE, STR_URL
from .languages import (
    ISO_LANGUAGE_SCRIPT,
    check_language_code,
)

if TYPE_CHECKING:
    from .models.sentence_transformer_wrapper import SentenceTransformerWrapper

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
]
DISTANCE_METRICS = Literal["cosine", "max_sim", "dot"]


def sentence_transformers_loader(
    model_name: str, revision: str | None = None, **kwargs
) -> SentenceTransformerWrapper:
    from .models.sentence_transformer_wrapper import SentenceTransformerWrapper

    return SentenceTransformerWrapper(model=model_name, revision=revision, **kwargs)


def get_loader_name(
    loader: Callable[..., Encoder] | None,
) -> str | None:
    if loader is None:
        return None
    if hasattr(loader, "func"):  # partial class wrapper
        return loader.func.__name__
    return loader.__name__


class ModelMeta(BaseModel):
    """The model metadata object.

    Attributes:
        loader: the function that loads the model. If None it will just default to loading the model using the sentence transformer library.
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
        training_datasets: A dictionary of datasets that the model was trained on. Names should be names as they appear in `mteb` for example
            {"ArguAna": ["test"]} if the model is trained on the ArguAna test set. This field is used to determine if a model generalizes zero-shot to
            a benchmark as well as mark dataset contaminations.
        adapted_from: Name of the model from which this model is adapted. For quantizations, fine-tunes, long doc extensions, etc.
        superseded_by: Name of the model that supersedes this model, e.g., nvidia/NV-Embed-v2 supersedes v1.
        is_cross_encoder: Whether the model can act as a cross-encoder or not.
        modalities: A list of strings representing the modalities the model supports. Default is ["text"].
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None
    revision: str | None
    release_date: STR_DATE | None
    languages: list[ISO_LANGUAGE_SCRIPT] | None
    loader: Callable[..., Encoder] | None = None
    n_parameters: int | None
    memory_usage_mb: float | None
    max_tokens: float | None
    embed_dim: int | None
    license: LICENSES | STR_URL | None
    open_weights: bool | None
    public_training_code: str | None
    public_training_data: str | bool | None
    framework: list[FRAMEWORKS]
    reference: STR_URL | None = None
    similarity_fn_name: DISTANCE_METRICS | None
    use_instructions: bool | None
    training_datasets: dict[str, list[str]] | None
    adapted_from: str | None = None
    superseded_by: str | None = None
    is_cross_encoder: bool | None = None
    modalities: list[MODALITIES] = ["text"]

    def to_dict(self):
        dict_repr = self.model_dump()
        loader = dict_repr.pop("loader", None)
        dict_repr["loader"] = get_loader_name(loader)
        return dict_repr

    @field_validator("languages")
    @classmethod
    def languages_are_valid(cls, languages: list[ISO_LANGUAGE_SCRIPT] | None) -> None:
        if languages is None:
            return None

        for code in languages:
            check_language_code(code)
        return languages

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str | None) -> str | None:
        if v is None or v == "bm25s":
            return v
        if "/" not in v:
            raise ValueError(
                "Model name must be in the format 'organization/model_name'"
            )
        return v

    def load_model(self, **kwargs: Any) -> Encoder:
        if self.loader is None:
            logger.warning(
                f"Loader not specified for model {self.name}, loading using sentence transformers."
            )
            loader = partial(
                sentence_transformers_loader,
                model_name=self.name,
                revision=self.revision,
                **kwargs,
            )
        else:
            loader = self.loader

        model: Encoder = loader(**kwargs)  # type: ignore
        model.mteb_model_meta = self
        return model

    def model_name_as_path(self) -> str:
        if self.name is None:
            raise ValueError("Model name is not set")
        return self.name.replace("/", "__").replace(" ", "_")

    def is_zero_shot_on(self, tasks: Sequence[AbsTask] | Sequence[str]) -> bool | None:
        """Indicates whether the given model can be considered
        zero-shot or not on the given tasks.
        Returns None if no training data is specified on the model.
        """
        # If no tasks were specified, we're obviously zero-shot
        if not tasks:
            return True
        training_datasets = self.get_training_datasets()
        # If no tasks were specified, we're obviously zero-shot
        if training_datasets is None:
            return None
        model_datasets = {ds_name for ds_name, splits in training_datasets.items()}
        if isinstance(tasks[0], str):
            benchmark_datasets = set(tasks)
        else:
            tasks = cast(Sequence[AbsTask], tasks)
            benchmark_datasets = set()
            for task in tasks:
                benchmark_datasets.add(task.metadata.name)
        intersection = model_datasets & benchmark_datasets
        return len(intersection) == 0

    def get_training_datasets(self) -> dict[str, list[str]] | None:
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
            except ValueError as e:
                logger.warning(f"Could not get source model: {e} in MTEB")

        return_dataset = training_datasets.copy()
        visited = set()

        for dataset in training_datasets:
            similar_tasks = collect_similar_tasks(dataset, visited)
            return_dataset |= {task: [] for task in similar_tasks}

        return return_dataset

    def zero_shot_percentage(
        self, tasks: Sequence[AbsTask] | Sequence[str]
    ) -> int | None:
        """Indicates how out-of-domain the selected tasks are for the given model."""
        training_datasets = self.get_training_datasets()
        if (training_datasets is None) or (not tasks):
            return None
        model_datasets = {ds_name for ds_name, splits in training_datasets.items()}
        if isinstance(tasks[0], str):
            benchmark_datasets = set(tasks)
        else:
            tasks = cast(Sequence[AbsTask], tasks)
            benchmark_datasets = {task.metadata.name for task in tasks}
        overlap = model_datasets & benchmark_datasets
        perc_overlap = 100 * (len(overlap) / len(benchmark_datasets))
        return int(100 - perc_overlap)

    def calculate_memory_usage_mb(self) -> int | None:
        """Calculates the memory usage (in FP32) of the model in MB."""
        if "API" in self.framework:
            return None

        MB = 1024**2
        try:
            safetensors_metadata = get_safetensors_metadata(self.name)
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


def collect_similar_tasks(dataset: str, visited: set[str]) -> set[str]:
    """Recursively collect all similar tasks for a given dataset."""
    from .overview import SIMILAR_TASKS

    if dataset in visited:
        return set()

    visited.add(dataset)
    similar = set()

    # Check if dataset is a key in SIMILAR_TASKS
    if dataset in SIMILAR_TASKS:
        for similar_task in SIMILAR_TASKS[dataset]:
            similar.add(similar_task)
            similar.update(collect_similar_tasks(similar_task, visited))

    # Check if dataset appears as a value in SIMILAR_TASKS
    for parent, children in SIMILAR_TASKS.items():
        if dataset in children:
            similar.add(parent)
            similar.update(collect_similar_tasks(parent, visited))

    return similar
