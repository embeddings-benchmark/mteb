from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Any

import numpy as np
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class SeedTextEmbeddingModel(Wrapper):
    def __init__(
        self,
        model_name: str,
        rate_limit_per_minute: int = 300,
        **kwargs,
    ) -> None:
        requires_package(
            self,
            "openai",
            model_name,
            install_instruction="pip install 'mteb[openai]'",
        )
        from openai import OpenAI

        requires_package(
            self,
            "tiktoken",
            model_name,
            install_instruction="pip install 'mteb[openai]'",
        )
        import tiktoken

        self.model_name = model_name
        self.rate_limit_per_minute = rate_limit_per_minute
        self.last_request_time = 0
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.client = OpenAI(
            api_key=os.environ["ARK_API_KEY"],
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit_per_minute

        if time_since_last_request < min_interval:
            time.sleep(min_interval - time_since_last_request)

        self.last_request_time = time.time()

    def _truncate_text(self, text: str, max_tokens: int = 32000) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens)
        return text

    def _format_instruction(self, instruction: str, input_: str) -> str:
        if isinstance(instruction, dict):
            return input_
        elif isinstance(instruction, str) and len(instruction):
            return instruction + "\n" + input_
        else:
            return input_

    def _embed(
        self,
        sentences: list[str],
        instruction: str,
        show_progress_bar: bool = False,
        retries: int = 5,
    ) -> np.ndarray:
        max_batch_size = 20
        batches = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm.tqdm(batches, leave=False, disable=not show_progress_bar):
            # Truncate texts
            batch = [self._truncate_text(text) for text in batch]

            # Add instruction to each text
            batch = [self._format_instruction(instruction, text) for text in batch]

            while retries > 0:
                try:
                    self._enforce_rate_limit()
                    response = self.client.embeddings.create(
                        model=self.model_name, input=batch, encoding_format="float"
                    )
                    embeddings = [x.embedding for x in response.data]
                    break
                except Exception as e:
                    logger.warning(
                        f"Retrying... {retries} retries left. Error: {str(e)}"
                    )
                    retries -= 1
                    if retries == 0:
                        raise e

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        logger.warning("The API will be publicly available soon. Stay tuned!")

        instruction = self.get_instruction(task_name, prompt_type)
        show_progress_bar = kwargs.pop("show_progress_bar", False)

        return self._embed(
            sentences,
            instruction=instruction,
            show_progress_bar=show_progress_bar,
        )


seed_embedding = ModelMeta(
    name="ByteDance-Seed/Doubao-1.5-Embedding",
    revision="2",
    release_date="2025-04-25",
    languages=[
        "eng-Latn",
        "zho-Hans",
    ],
    loader=partial(
        SeedTextEmbeddingModel,
        model_name="doubao-1-5-embedding",
        rate_limit_per_minute=300,
    ),
    max_tokens=32768,
    embed_dim=2048,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://huggingface.co/ByteDance-Seed/Doubao-1.5-Embedding",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)
