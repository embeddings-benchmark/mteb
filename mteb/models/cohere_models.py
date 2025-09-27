from __future__ import annotations

from functools import partial
from typing import Any, Literal, get_args

import numpy as np
import torch
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper

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

EMBEDDING_TYPE = Literal[
    "float",
    "int8",
    "uint8",
    "binary",
]


# Implementation follows https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/main/src/seb/registered_models/cohere_models.py
class CohereTextEmbeddingModel(Wrapper):
    def __init__(
        self,
        model_name: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        embedding_type: EMBEDDING_TYPE = "float",
        output_dimension: int | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.sep = sep
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)
        assert embedding_type in get_args(EMBEDDING_TYPE)
        self.embedding_type = embedding_type
        self.output_dimension = output_dimension

    def _embed(
        self,
        sentences: list[str],
        cohere_task_type: str,
        show_progress_bar: bool = False,
        retries: int = 5,
    ) -> torch.Tensor:
        import cohere  # type: ignore

        max_batch_size = 256

        batches = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        client = cohere.Client()

        all_embeddings = []

        for batch in tqdm.tqdm(batches, leave=False, disable=not show_progress_bar):
            while retries > 0:  # Cohere's API is not always reliable
                try:
                    embed_kwargs = {
                        "texts": batch,
                        "model": self.model_name,
                        "input_type": cohere_task_type,
                        "embedding_types": [self.embedding_type],
                    }
                    if self.output_dimension is not None:
                        embed_kwargs["output_dimension"] = self.output_dimension

                    response = client.embed(**embed_kwargs)
                    break
                except Exception as e:
                    print(f"Retrying... {retries} retries left.")
                    retries -= 1
                    if retries == 0:
                        raise e

            # Get embeddings based on requested type
            if self.embedding_type == "float":
                embeddings = response.embeddings.float
            elif self.embedding_type == "int8":
                embeddings = response.embeddings.int8
            elif self.embedding_type == "uint8":
                embeddings = response.embeddings.uint8
            elif self.embedding_type == "binary":
                embeddings = response.embeddings.binary
            else:
                raise ValueError(f"Embedding type {self.embedding_type} not allowed")
            all_embeddings.extend(torch.tensor(embeddings).numpy())

        embeddings_array = np.array(all_embeddings)

        # Post-process embeddings based on type (similar to voyage_models.py)
        primary_embedding_type = self.embedding_type

        if primary_embedding_type == "binary":
            # Unpack bit-packed embeddings: each byte contains 8 embedding values
            unpacked_embeddings = []
            for embedding in embeddings_array:
                # Convert bytes to bits and unpack
                unpacked = []
                for byte_val in embedding:
                    # Extract 8 bits from each byte (LSB first)
                    for bit_pos in range(8):
                        bit_val = (byte_val >> bit_pos) & 1
                        # Convert 0/1 to -1/1 for binary (signed)
                        unpacked.append(1.0 if bit_val else -1.0)
                unpacked_embeddings.append(unpacked)
            embeddings_array = np.array(unpacked_embeddings, dtype=np.float32)
        elif primary_embedding_type in ["int8", "uint8"]:
            # Convert int8/uint8 embeddings to float32
            embeddings_array = embeddings_array.astype(np.float32)

        return embeddings_array

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


model_prompts = {
    "Classification": "classification",
    "MultilabelClassification": "classification",
    "Clustering": "clustering",
    PromptType.query.value: "search_query",
    PromptType.document.value: "search_document",
}

cohere_mult_3 = ModelMeta(
    loader=partial(  # type: ignore
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
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=512,
    reference="https://cohere.com/blog/introducing-embed-v3",
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
)

cohere_eng_3 = ModelMeta(
    loader=partial(  # type: ignore
        CohereTextEmbeddingModel,
        model_name="embed-english-v3.0",
        model_prompts=model_prompts,
    ),
    name="Cohere/Cohere-embed-english-v3.0",
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
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
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=384,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
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
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=384,
    license=None,
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
)
