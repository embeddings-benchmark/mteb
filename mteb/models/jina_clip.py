from __future__ import annotations

from functools import partial
<<<<<<< HEAD
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from mteb.model_meta import ModelMeta
# from mteb.models.text_formatting_utils import corpus_to_texts


class JinaCLIPModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )

    def get_text_embeddings(self, texts: list[str], batch_size: int = 32):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                text_outputs = self.model.encode_text(
                    batch_texts, convert_to_numpy=False, convert_to_tensor=True
                )
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(
        self, images: list[Image.Image] | DataLoader, batch_size: int = 32
    ):
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    image_outputs = self.model.encode_image(
                        batch, convert_to_numpy=False, convert_to_tensor=True
                    )
                    all_image_embeddings.append(image_outputs.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    image_outputs = self.model.encode_image(
                        batch_images, convert_to_numpy=False, convert_to_tensor=True
                    )
                    all_image_embeddings.append(image_outputs.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] = None,
        fusion_mode="sum",
        batch_size: int = 32,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.encode_text(
                texts,
                batch_size=batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
            )

        if images is not None:
            image_embeddings = self.encode_image(
                images,
                batch_size=batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
            )

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            if fusion_mode == "sum":
                fused_embeddings = text_embeddings + image_embeddings
            else:
                # to do: add other fusion mode
                raise ValueError(f"fusion mode {fusion_mode} hasn't been implemented")
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        if "prompt_name" in kwargs:
            kwargs.pop("prompt_name")
        return self.model.encode_text(sentences, batch_size=batch_size, **kwargs)

    # def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
    #     if "prompt_name" in kwargs:
    #         kwargs.pop("prompt_name")
    #     sentences = [
    #         "Represent this sentence for searching relevant passages: " + sentence
    #         for sentence in queries
    #     ]
    #     emb = self.encode(
    #         sentences, batch_size=batch_size, normalize_embeddings=True, **kwargs
    #     )
    #     return emb

    # def encode_corpus(
    #     self,
    #     corpus: list[dict[str, str]] | dict[str, list[str]],
    #     batch_size: int = 32,
    #     **kwargs: Any,
    # ):
    #     if "prompt_name" in kwargs:
    #         kwargs.pop("prompt_name")
    #     sentences = corpus_to_texts(corpus)
    #     emb = self.encode(
    #         sentences, batch_size=batch_size, normalize_embeddings=True, **kwargs
    #     )
    #     return emb
=======
>>>>>>> mieb

from mteb.model_meta import ModelMeta, sentence_transformers_loader

jina_clip_v1 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="jinaai/jina-clip-v1",
    ),
    name="jinaai/jina-clip-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision="1cbe5e8b11ea3728df0b610d5453dfe739804aa9",
    release_date="2024-05-30",
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(jina_clip_v1.name, jina_clip_v1.revision)
    emb = mdl.get_text_embeddings(["Hello, world!"])
