from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies


class JinaCLIPModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_image_dependencies()

        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        convert_to_numpy=False,
        convert_to_tensor=True,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                text_outputs = self.model.encode_text(
                    batch_texts,
                    convert_to_numpy=convert_to_numpy,
                    convert_to_tensor=convert_to_tensor,
                )
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        convert_to_numpy=False,
        convert_to_tensor=True,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    image_outputs = self.model.encode_image(
                        [F.to_pil_image(b.to("cpu")) for b in batch],
                        convert_to_numpy=convert_to_numpy,
                        convert_to_tensor=convert_to_tensor,
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
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(
                texts, convert_to_numpy=False, convert_to_tensor=True, **kwargs
            )

        if images is not None:
            image_embeddings = self.get_image_embeddings(
                images, convert_to_numpy=False, convert_to_tensor=True, **kwargs
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
        if "task_name" in kwargs:
            kwargs.pop("task_name")
        return self.model.encode_text(sentences, batch_size=batch_size, **kwargs)


jina_clip_v1 = ModelMeta(
    loader=partial(
        JinaCLIPModelWrapper,
        model_name="jinaai/jina-clip-v1",
    ),
    name="jinaai/jina-clip-v1",
    languages=["eng-Latn"],
    revision="06150c7c382d7a4faedc7d5a0d8cdb59308968f4",
    release_date="2024-05-30",
    modalities=["image", "text"],
    n_parameters=223_000_000,
    memory_usage_mb=849,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/jinaai/jina-clip-v1",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # LAION400M
        # ShareGPT4V
        "MSMARCO": ["train"],
        # NQ
        # HotpotQA
        # Natural Language Inference (NLI) dataset (Bowman et al., 2015)
    },
)
