from __future__ import annotations

from datetime import date
from functools import partial
from typing import Any, Callable, Literal

from pydantic import BaseModel, BeforeValidator, TypeAdapter
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode

from .languages import ISO_LANGUAGE_SCRIPT

Frameworks = Literal["Sentence Transformers", "PyTorch"]

pastdate_adapter = TypeAdapter(date)
STR_DATE = Annotated[
    str, BeforeValidator(lambda value: str(pastdate_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a valid date


def sentence_transformers_loader(model_name: str, revision: str) -> SentenceTransformer:
    return SentenceTransformer(model_name_or_path=model_name, revision=revision)


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
        revision: The revision number of the model.
        release_date: The date the model's revision was released.
        license: The license under which the model is released. Required if open_source is True.
        open_source: Whether the model is open source or proprietary.
        framework: The framework the model is implemented in, can be a list of frameworks e.g. `["Sentence Transformers", "PyTorch"]`.
        languages: The languages the model is intended for specified as a 3 letter language code followed by a script code e.g. "eng-Latn" for English
            in the Latin script.
    """

    name: str
    revision: str
    release_date: STR_DATE
    languages: list[ISO_LANGUAGE_SCRIPT]
    loader: Callable[..., Encoder | EncoderWithQueryCorpusEncode] | None = None
    n_parameters: int | None = None
    memory_usage: float | None = None
    max_tokens: int | None = None
    embed_dim: int | None = None
    license: str | None = None
    open_source: bool | None = None
    framework: list[Frameworks] = []

    def to_dict(self):
        dict_repr = self.model_dump()
        loader = dict_repr.pop("loader", None)
        dict_repr["loader"] = loader.__name__ if loader is not None else None
        return dict_repr

    def load_model(self, **kwargs: Any) -> Encoder | EncoderWithQueryCorpusEncode:
        if self.loader is None:
            loader = partial(
                sentence_transformers_loader,
                model_name=self.name,
                revision=self.revision,
            )
        else:
            loader = self.loader

        model: Encoder | EncoderWithQueryCorpusEncode = loader(**kwargs)  # type: ignore
        return model
