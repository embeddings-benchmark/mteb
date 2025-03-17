from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.encoder_interface import BatchedInput, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class ALIGNModelWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                inputs = self.processor(
                    text=batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.get_text_features(**inputs)
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs,
    ):
        all_image_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                inputs = self.processor(
                    images=batch["image"], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_outputs = self.model.get_image_features(**inputs)
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

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        fusion_mode: Literal["sum"] = "sum",
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
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


align_base = ModelMeta(
    loader=ALIGNModelWrapper,
    name="kakaobrain/align-base",
    languages=["eng_Latn"],
    revision="e96a37facc7b1f59090ece82293226b817afd6ba",
    release_date="2023-02-24",
    modalities=["image", "text"],
    n_parameters=176_000_000,
    memory_usage_mb=671,
    max_tokens=64,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code="https://github.com/kakaobrain/coyo-align",
    public_training_data=True,
    framework=["PyTorch"],
    reference="https://huggingface.co/kakaobrain/align-base",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # COYO-700M
    },
)
