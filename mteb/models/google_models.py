"""Required python >=3.9 and that the `google-generativeai` package is installed. Additionally the GOOGLE_API_KEY environment variable must be set."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta


class GoogleTextEmbeddingModel(Encoder):
    def __init__(self, model_name: str, sep: str = " ", **kwargs) -> None:
        self.model_name = model_name

    def _embed(
        self, sentences: list[str], *, task_type: str, titles: list[str] | None = None
    ) -> np.ndarray:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "`google-generativeai` is required to run the google API, please install it using `pip install google-generativeai`"
            )

        if titles:
            result = genai.embed_content(  # type: ignore
                model=self.model_name,
                content=sentences,
                task_type=task_type,
                title=titles,
            )
        else:
            result = genai.embed_content(  # type: ignore
                model=self.model_name,
                content=sentences,
                task_type=task_type,
            )

        return np.asarray(result["embedding"])

    def encode(
        self,
        sentences: list[str],
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        input_type = None  # Default
        if prompt_name:
            task = mteb.get_task(prompt_name)
            task_type = task.metadata.type
            if task_type in ["Classification", "MultilabelClassification"]:
                input_type = "CLASSIFICATION"
            elif task_type == "Clustering":
                input_type = "CLUSTERING"
            elif task_type == "STS":
                input_type = "SIMILARITY"
        return self._embed(sentences, task_type=input_type)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self._embed(queries, task_type="RETRIEVAL_QUERY")

    def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:
        if isinstance(corpus, dict):
            sentences, titles = [], []

            for i in range(len(corpus["text"])):  # type: ignore
                titles.append(corpus["title"][i])  # type: ignore
                sentences.append(corpus["text"][i])  # type: ignore
        else:
            sentences, titles = [], []
            for doc in corpus:
                titles.append(doc["title"])
                sentences.append(doc["text"])
        return self._embed(sentences, task_type="RETRIEVAL_DOCUMENT")


name = "models/text-embedding-004"
google_emb_004 = ModelMeta(
    loader=partial(GoogleTextEmbeddingModel, model_name=name),
    name=name,
    languages=["eng-Latn"],
    open_source=False,
    revision="1",  # revision is intended for implementation
    release_date=None,  # couldnt figure this out
    n_parameters=None,
    memory_usage=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    similarity_fn_name="cosine",  # assumed
    framework=[],
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(google_emb_004.name, google_emb_004.revision)
    emb = mdl.encode(["Hello, world!"])
