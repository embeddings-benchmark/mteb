from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

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
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
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
        cohere_task_type = self.get_prompt_name(
            self.model_prompts, task_name, prompt_type
        )
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
    name="Cohere/Cohere-embed-multilingual-v3.0",
    languages=supported_languages,
    open_weights=False,
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=None,
    embed_dim=512,
    reference="https://cohere.com/blog/introducing-embed-v3",
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)

cohere_eng_3 = ModelMeta(
    loader=partial(
        CohereTextEmbeddingModel,
        model_name="embed-multilingual-v3.0",
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-english-v3.0",
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
    use_instructions=False,
)


cohere_mult_light_3 = ModelMeta(
    loader=partial(
        CohereTextEmbeddingModel,
        model_name="embed-multilingual-light-v3.0",
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-multilingual-light-v3.0",
    languages=supported_languages,
    open_weights=False,
    revision="1",
    reference="https://cohere.com/blog/introducing-embed-v3",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=512,
    embed_dim=384,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)

cohere_eng_light_3 = ModelMeta(
    loader=partial(
        CohereTextEmbeddingModel,
        model_name="embed-english-light-v3.0",
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-english-light-v3.0",
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage=None,
    max_tokens=512,
    embed_dim=384,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)
