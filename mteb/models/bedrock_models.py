from __future__ import annotations

import json
import logging
import re
from functools import partial
from typing import Any

import numpy as np
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.cohere_models import model_prompts as cohere_model_prompts
from mteb.models.cohere_models import supported_languages as cohere_supported_languages
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class BedrockWrapper(Wrapper):
    def __init__(
        self,
        model_id: str,
        provider: str,
        max_tokens: int,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "boto3", "The AWS SDK for Python")
        import boto3

        boto3_session = boto3.session.Session()
        region_name = boto3_session.region_name
        self._client = boto3.client("bedrock-runtime", region_name)

        self._model_id = model_id
        self._provider = provider.lower()

        if self._provider == "cohere":
            self.model_prompts = (
                self.validate_task_to_prompt_name(model_prompts)
                if model_prompts
                else None
            )
            self._max_batch_size = 96
            self._max_sequence_length = max_tokens * 4
        else:
            self._max_tokens = max_tokens

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        requires_package(self, "boto3", "Amazon Bedrock")
        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )
        if self._provider == "amazon":
            return self._encode_amazon(sentences, show_progress_bar)
        elif self._provider == "cohere":
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
            cohere_task_type = self.model_prompts.get(prompt_name, "search_document")
            return self._encode_cohere(sentences, cohere_task_type, show_progress_bar)
        else:
            raise ValueError(
                f"Unknown provider '{self._provider}'. Must be 'amazon' or 'cohere'."
            )

    def _encode_amazon(
        self, sentences: list[str], show_progress_bar: bool = False
    ) -> np.ndarray:
        from botocore.exceptions import ValidationError

        all_embeddings = []
        # https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
        max_sequence_length = int(self._max_tokens * 4.5)

        for sentence in tqdm.tqdm(
            sentences, leave=False, disable=not show_progress_bar
        ):
            if len(sentence) > max_sequence_length:
                truncated_sentence = sentence[:max_sequence_length]
            else:
                truncated_sentence = sentence

            try:
                embedding = self._embed_amazon(truncated_sentence)
                all_embeddings.append(embedding)

            except ValidationError as e:
                error_str = str(e)
                pattern = r"request input token count:\s*(\d+)"
                match = re.search(pattern, error_str)
                if match:
                    num_tokens = int(match.group(1))

                    ratio = 0.9 * (self._max_tokens / num_tokens)
                    dynamic_cutoff = int(len(truncated_sentence) * ratio)

                    embedding = self._embed_amazon(truncated_sentence[:dynamic_cutoff])
                    all_embeddings.append(embedding)
                else:
                    raise e

        return np.array(all_embeddings)

    def _encode_cohere(
        self,
        sentences: list[str],
        cohere_task_type: str,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        batches = [
            sentences[i : i + self._max_batch_size]
            for i in range(0, len(sentences), self._max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm.tqdm(batches, leave=False, disable=not show_progress_bar):
            response = self._client.invoke_model(
                body=json.dumps(
                    {
                        "texts": [sent[: self._max_sequence_length] for sent in batch],
                        "input_type": cohere_task_type,
                    }
                ),
                modelId=self._model_id,
                accept="*/*",
                contentType="application/json",
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)

    def _embed_amazon(self, sentence: str) -> np.ndarray:
        response = self._client.invoke_model(
            body=json.dumps({"inputText": sentence}),
            modelId=self._model_id,
            accept="application/json",
            contentType="application/json",
        )
        return self._to_numpy(response)

    def _to_numpy(self, embedding_response) -> np.ndarray:
        response = json.loads(embedding_response.get("body").read())
        key = "embedding" if self._provider == "amazon" else "embeddings"
        return np.array(response[key])


amazon_titan_embed_text_v1 = ModelMeta(
    name="bedrock/amazon-titan-embed-text-v1",
    revision="1",
    release_date="2023-09-27",
    languages=None,  # not specified
    loader=partial(
        BedrockWrapper,
        model_id="amazon.titan-embed-text-v1",
        provider="amazon",
        max_tokens=8192,
    ),
    max_tokens=8192,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    license=None,
    reference="https://aws.amazon.com/about-aws/whats-new/2023/09/amazon-titan-embeddings-generally-available/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)

amazon_titan_embed_text_v2 = ModelMeta(
    name="bedrock/amazon-titan-embed-text-v2",
    revision="1",
    release_date="2024-04-30",
    languages=None,  # not specified
    loader=partial(
        BedrockWrapper,
        model_id="amazon.titan-embed-text-v2:0",
        provider="amazon",
        max_tokens=8192,
    ),
    max_tokens=8192,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    license=None,
    reference="https://aws.amazon.com/about-aws/whats-new/2024/04/amazon-titan-text-embeddings-v2-amazon-bedrock/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)
# Note: For the original Cohere API implementation, refer to:
# https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/cohere_models.py
# This implementation uses the Amazon Bedrock endpoint for Cohere models.
cohere_embed_english_v3 = ModelMeta(
    loader=partial(
        BedrockWrapper,
        model_id="cohere.embed-english-v3",
        provider="cohere",
        max_tokens=512,
        model_prompts=cohere_model_prompts,
    ),
    name="bedrock/cohere-embed-english-v3",
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
)

cohere_embed_multilingual_v3 = ModelMeta(
    loader=partial(
        BedrockWrapper,
        model_id="cohere.embed-multilingual-v3",
        provider="cohere",
        max_tokens=512,
        model_prompts=cohere_model_prompts,
    ),
    name="bedrock/cohere-embed-multilingual-v3",
    languages=cohere_supported_languages,
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
)
