import asyncio

import numpy as np
from typing import Any
from openai import AsyncOpenAI

import mteb
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.models.instructions import task_to_instruction

MODEL = "intfloat/e5-mistral-7b-instruct"
TASKS = ["STS12"]

class AsyncOpenAIWrapper:
    def __init__(self, 
                 model_name: str, 
                 base_url: str = "http://localhost:8000/v1",
                 num_concurrent: int = 100) -> None:        

        self._client = AsyncOpenAI(base_url=base_url)
        self._model_name = model_name
        self._num_concurrent = num_concurrent

    async def async_encode(self, sentences: list[str]) -> np.ndarray:
        sublists = [
            sentences[i : i + self._num_concurrent]
            for i in range(0, len(sentences), self._num_concurrent)
        ]

        all_embeddings = []

        for sublist in sublists:
            outputs = [
                self._client.embeddings.create(
                    input=sentence,
                    model=self._model_name
                ) for sentence in sublist
            ]
            for output in await asyncio.gather(*outputs):
                all_embeddings.extend(self._to_numpy(output))

        return np.array(all_embeddings)

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        def e5_instruction(instruction: str) -> str:
            return f"Instruct: {instruction}\nQuery: "
        
        if "prompt_name" in kwargs:
            if "instruction" in kwargs:
                raise ValueError(
                    "Cannot specify both `prompt_name` and `instruction`."
                )
            instruction = task_to_instruction(
                kwargs.pop("prompt_name"), kwargs.pop("is_query", True)
            )
        else:
            instruction = kwargs.pop("instruction", "")
        if instruction:
            formatted_instruction = e5_instruction(instruction)
            sentences = [
                f"{formatted_instruction}{sentence}" for sentence in sentences]

        return asyncio.run(self.async_encode(sentences))

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list[str]], **kwargs: Any
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus)
        return self.encode(sentences, **kwargs)

    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])



model = AsyncOpenAIWrapper(model_name=MODEL)
model.mteb_model_meta = ModelMeta(
    name=MODEL,
    revision=None,
    release_date=None,
    languages=None,
    similarity_fn_name=None,)

tasks = mteb.get_tasks(tasks=TASKS)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/")
