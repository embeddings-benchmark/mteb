from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch

import mteb
from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta


# Implementation follows https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/main/src/seb/registered_models/cohere_models.py
class CohereTextEmbeddingModel(Encoder):
    def __init__(self, model_name: str, sep: str = " ", **kwargs) -> None:
        self.model_name = model_name
        self.sep = sep

    def _embed(
        self, sentences: list[str], input_type: str, retries: int = 5
    ) -> torch.Tensor:
        import cohere  # type: ignore

        client = cohere.Client()
        while retries > 0:  # Cohere's API is not always reliable
            try:
                response = client.embed(
                    texts=list(sentences),
                    model=self.model_name,
                    input_type=input_type,
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
        prompt_name: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> np.ndarray:
        input_type = "search_document"
        if prompt_name:
            task = mteb.get_task(prompt_name)
            task_type = task.metadata.type
            if task_type in ["Classification", "MultilabelClassification"]:
                input_type = "classification"
            elif task_type == "Clustering":
                input_type = "clustering"
        return self._embed(sentences, input_type=input_type).numpy()

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        return self._embed(queries, input_type="search_query").numpy()

    def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()  # type: ignore
                if "title" in corpus
                else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        return self._embed(sentences, input_type="search_document").numpy()


cohere_mult_3 = ModelMeta(
    loader=partial(CohereTextEmbeddingModel, model_name="embed-multilingual-v3.0"),
    name="embed-multilingual-v3.0",
    languages=[],  # Unknown, but support >100 languages
    open_source=False,
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
)

cohere_eng_3 = ModelMeta(
    loader=partial(CohereTextEmbeddingModel, model_name="embed-english-v3.0"),
    name="embed-english-v3.0",
    languages=["eng-Latn"],
    open_source=False,
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
)

if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(cohere_mult_3.name, cohere_mult_3.revision)
    emb = mdl.encode(["Hello, world!"])
