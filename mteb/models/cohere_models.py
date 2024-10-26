from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import (
    get_prompt_name,
    validate_task_to_prompt_name,
)

from .wrapper import Wrapper


# Implementation follows https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/main/src/seb/registered_models/cohere_models.py
class CohereTextEmbeddingModel(Wrapper):
    def __init__(
        self,
        model_name: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.sep = sep
        self.model_prompts = (
            validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def _embed(
        self, sentences: list[str], cohere_task_type: str, retries: int = 5
    ) -> torch.Tensor:
        import cohere  # type: ignore

        client = cohere.Client()
        while retries > 0:  # Cohere's API is not always reliable
            try:
                response = client.embed(
                    texts=list(sentences),
                    model=self.model_name,
                    input_type=cohere_task_type,
                )
                break
            except Exception as e:
                print(f"Retrying... {retries} retries left.")
                retries -= 1
                if retries == 0:
                    raise e
        return torch.tensor(response.embeddings)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        cohere_task_type = get_prompt_name(self.model_prompts, task_name, prompt_type)
        if cohere_task_type is None:
            # search_document is recommended if unknown (https://cohere.com/blog/introducing-embed-v3)
            cohere_task_type = "search_document"
        return self._embed(sentences, cohere_task_type=cohere_task_type).numpy()


model_prompts = {
    "Classification": "classification",
    "MultilabelClassification": "classification",
    "Clustering": "clustering",
    PromptType.query.value: "search_query",
    PromptType.passage.value: "search_document",
}

cohere_mult_3 = ModelMeta(
    loader=partial(
        CohereTextEmbeddingModel,
        model_name="embed-multilingual-v3.0",
        model_prompts=model_prompts,
    ),
    name="embed-multilingual-v3.0",
    languages=[],  # Unknown, but support >100 languages
    open_weights=False,
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instuctions=False,
)

cohere_eng_3 = ModelMeta(
    loader=partial(
        CohereTextEmbeddingModel,
        model_name="embed-multilingual-v3.0",
        model_prompts=model_prompts,
    ),
    name="embed-english-v3.0",
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instuctions=False,
)
