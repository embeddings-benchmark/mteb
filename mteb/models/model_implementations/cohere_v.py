from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

all_languages = [
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


def cohere_v_loader(model_name, **kwargs):
    requires_package(
        cohere_v_loader, "cohere", model_name, "pip install 'mteb[cohere]'"
    )
    import cohere

    class CohereMultiModalModelWrapper(AbsEncoder):
        def __init__(
            self,
            model_name: str,
            **kwargs: Any,
        ):
            """Wrapper for Cohere multimodal embedding model,

            do `export COHERE_API_KEY=<Your_Cohere_API_KEY>` before running eval scripts.
            Cohere currently supports 40 images/min, thus time.sleep(1.5) is applied after each image.
            Remove or adjust this after Cohere API changes capacity.
            """
            requires_image_dependencies()
            from torchvision import transforms

            self.model_name = model_name
            api_key = os.getenv("COHERE_API_KEY")
            self.client = cohere.ClientV2(api_key)
            self.image_format = "JPEG"
            self.transform = transforms.Compose([transforms.PILToTensor()])

        def get_text_embeddings(
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                response = self.client.embed(
                    texts=batch["text"],
                    model=self.model_name,
                    input_type="search_document",
                )
                all_text_embeddings.append(torch.tensor(response.embeddings.float))

            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                for image in batch:
                    # cohere only supports 1 image per call
                    buffered = io.BytesIO()
                    image = self.transform(image)
                    image.save(buffered, format=self.image_format)
                    image_bytes = buffered.getvalue()
                    stringified_buffer = base64.b64encode(image_bytes).decode("utf-8")
                    content_type = f"image/{self.image_format.lower()}"
                    image_base64 = f"data:{content_type};base64,{stringified_buffer}"
                    response = self.client.embed(
                        model=self.model_name,
                        input_type="image",
                        embedding_types=["float"],
                        images=[image_base64],
                    )
                    all_image_embeddings.append(torch.tensor(response.embeddings.float))
                    time.sleep(1.5)
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            text_embeddings = None
            image_embeddings = None
            if "text" in inputs.dataset.features:
                text_embeddings = self.get_text_embeddings(inputs, **kwargs)
            if "image" in inputs.dataset.features:
                image_embeddings = self.get_image_embeddings(inputs, **kwargs)

            if text_embeddings is not None and image_embeddings is not None:
                if len(text_embeddings) != len(image_embeddings):
                    raise ValueError(
                        "The number of texts and images must have the same length"
                    )
                fused_embeddings = text_embeddings + image_embeddings
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings
            raise ValueError

    return CohereMultiModalModelWrapper(model_name, **kwargs)


cohere_mult_3 = ModelMeta(
    loader=cohere_v_loader,  # type: ignore
    loader_kwargs={"model_name": "embed-multilingual-v3.0"},
    name="cohere/embed-multilingual-v3.0",
    languages=[],  # Unknown, but support >100 languages
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0",
    use_instructions=False,
    training_datasets=None,
)

cohere_eng_3 = ModelMeta(
    loader=cohere_v_loader,  # type: ignore
    loader_kwargs={"model_name": "embed-english-v3.0"},
    name="cohere/embed-english-v3.0",
    languages=["eng-Latn"],
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/Cohere/Cohere-embed-english-v3.0",
    use_instructions=False,
    training_datasets=None,
)

cohere_embed_v4_multimodal = ModelMeta(
    loader=cohere_v_loader,
    loader_kwargs=dict(model_name="embed-v4.0"),
    name="Cohere/Cohere-embed-v4.0",
    languages=all_languages,
    revision="1",
    release_date="2024-12-01",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=128000,
    embed_dim=1536,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://docs.cohere.com/docs/cohere-embed",
    use_instructions=False,
    training_datasets=None,
)
