from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel

from mteb.abstasks.TaskMetadata import STR_DATE, STR_URL
from mteb.encoder_interface import Encoder

from .languages import ISO_LANGUAGE_SCRIPT

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
]
DISTANCE_METRICS = Literal["cosine"]


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
        name: The name of the model, ideally the name on huggingface.
        n_parameters: The number of parameters in the model, e.g. 7_000_000 for a 7M parameter model. Can be None if the the number of parameters is not known (e.g. for proprietary models) or
            if the loader returns a SentenceTransformer model from which it can be derived.
        memory_usage: The amount of memory the model uses in GB. Can be None if the memory usage is not known (e.g. for proprietary models).
        max_tokens: The maximum number of tokens the model can handle. Can be None if the maximum number of tokens is not known (e.g. for proprietary
            models).
        embed_dim: The dimension of the embeddings produced by the model. Currently all models are assumed to produce fixed-size embeddings.
        revision: The revision number of the model. If None it is assumed that the metadata (including the loader) is valid for all revisions of the model.
        release_date: The date the model's revision was released.
        license: The license under which the model is released. Required if open_weights is True.
        open_weights: Whether the model is open source or proprietary.
        public_training_data: Whether the training data used to train the model is publicly available.
        public_training_code: Whether the code used to train the model is publicly available.
        similarity_fn_name: The distance metric used by the model.
        framework: The framework the model is implemented in, can be a list of frameworks e.g. `["Sentence Transformers", "PyTorch"]`.
        reference: A URL to the model's page on huggingface or another source.
        languages: The languages the model is intended for specified as a 3 letter language code followed by a script code e.g. "eng-Latn" for English
            in the Latin script.
        use_instuctions: Whether the model uses instructions E.g. for prompt-based models. This also include models that require a specific format for
            input such as "query: {document}" or "passage: {document}".
        zero_shot_benchmarks: A list of benchmarks on which the model has been evaluated in a zero-shot setting. By default we assume that all models
            are evaluated non-zero-shot unless specified otherwise.
    """

    name: str | None
    revision: str | None
    release_date: STR_DATE | None
    languages: list[ISO_LANGUAGE_SCRIPT] | None
    loader: Callable[..., Encoder] | None = None
    n_parameters: int | None = None
    memory_usage: float | None = None
    max_tokens: int | None = None
    embed_dim: int | None = None
    license: str | None = None
    open_weights: bool | None = None
    public_training_data: bool | None = None
    public_training_code: bool | None = None
    framework: list[FRAMEWORKS] = []
    reference: STR_URL | None = None
    similarity_fn_name: DISTANCE_METRICS | None = None
    use_instuctions: bool | None = None
    zero_shot_benchmarks: list[str] | None = None

    def to_dict(self):
        dict_repr = self.model_dump()
        loader = dict_repr.pop("loader", None)
        dict_repr["loader"] = get_loader_name(loader)
        return dict_repr

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
        return model

    def model_name_as_path(self) -> str:
        if self.name is None:
            raise ValueError("Model name is not set")
        return self.name.replace("/", "__").replace(" ", "_")
