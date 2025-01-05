from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any

import numpy as np
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_package

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


supported_languages = [
    "afr-Latn",
    "amh-Ethi",
    "ara-Arab",
    "asm-Beng",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "bod-Tibt",
    "bos-Latn",
    "cat-Latn",
    "ceb-Latn",
    "cos-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "epo-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "fry-Latn",
    "gle-Latn",
    "gla-Latn",
    "glg-Latn",
    "guj-Gujr",
    "hau-Latn",
    "haw-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hmn-Latn",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "ibo-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Kore",
    "kur-Arab",
    "kir-Cyrl",
    "lat-Latn",
    "ltz-Latn",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mlg-Latn",
    "mri-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mlt-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nya-Latn",
    "ori-Orya",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "kin-Latn",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "smo-Latn",
    "sna-Latn",
    "som-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "sot-Latn",
    "sun-Latn",
    "swe-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tgk-Cyrl",
    "tha-Thai",
    "tuk-Latn",
    "tgl-Latn",
    "tur-Latn",
    "tat-Cyrl",
    "uig-Arab",
    "ukr-Cyrl",
    "urd-Arab",
    "uzb-Latn",
    "vie-Latn",
    "wol-Latn",
    "xho-Latn",
    "yid-Hebr",
    "yor-Latn",
    "zho-Hans",
    "zul-Latn",
]


class CohereBedrockWrapper(Wrapper):
    def __init__(
        self, model_id: str, model_prompts: dict[str, str] | None = None, **kwargs
    ) -> None:
        requires_package(self, "boto3", "Amazon Bedrock")
        import boto3

        boto3_session = boto3.session.Session()
        region_name = boto3_session.region_name
        self._client = boto3.client(
            "bedrock-runtime",
            region_name,
        )
        self._model_id = model_id
        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def _embed(
        self,
        sentences: list[str],
        cohere_task_type: str,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        max_batch_size = 96

        batches = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm.tqdm(batches, leave=False, disable=not show_progress_bar):
            response = self._client.invoke_model(
                body=json.dumps(
                    {
                        "texts": [sent[:2048] for sent in batch],
                        "input_type": cohere_task_type,
                    }
                ),
                modelId=self._model_id,
                accept="*/*",
                contentType="application/json",
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        cohere_task_type = self.model_prompts.get(prompt_name)

        if cohere_task_type is None:
            # search_document is recommended if unknown (https://cohere.com/blog/introducing-embed-v3)
            cohere_task_type = "search_document"

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )

        return self._embed(
            sentences,
            cohere_task_type=cohere_task_type,
            show_progress_bar=show_progress_bar,
        )

    def _to_numpy(self, embedding_response) -> np.ndarray:
        response = json.loads(embedding_response.get("body").read())
        return np.array(response["embeddings"])


model_prompts = {
    "Classification": "classification",
    "MultilabelClassification": "classification",
    "Clustering": "clustering",
    PromptType.query.value: "search_query",
    PromptType.passage.value: "search_document",
}

cohere_embed_english_v3 = ModelMeta(
    loader=partial(
        CohereBedrockWrapper,
        model_id="cohere.embed-english-v3",
        model_prompts=model_prompts,
    ),
    name="bedrock/cohere-embed-english-v3",
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
)

cohere_embed_multilingual_v3 = ModelMeta(
    loader=partial(
        CohereBedrockWrapper,
        model_id="cohere.embed-multilingual-v3",
        model_prompts=model_prompts,
    ),
    name="cohere-embed-multilingual-v3",
    languages=supported_languages,
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
)
