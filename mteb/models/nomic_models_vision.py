from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


class NomicVisionModelWrapper:
    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.vision_model_name)
        self.vision_model = AutoModel.from_pretrained(
            self.vision_model_name, trust_remote_code=True
        ).to(self.device)
        self.text_model = AutoModel.from_pretrained(
            self.text_model_name, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)

        self.text_model.eval()
        self.vision_model.eval()

    def preprocess(
        self,
        texts: list[str],
        images: list[Image.Image],
    ):
        return self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.text_model(**inputs)
                text_embeddings = self.mean_pooling(
                    text_outputs, inputs["attention_mask"]
                )
                text_embeddings = F.layer_norm(
                    text_embeddings, normalized_shape=(text_embeddings.shape[1],)
                )
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                all_text_embeddings.append(text_embeddings.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    inputs = self.processor(images=batch, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.vision_model(**inputs).last_hidden_state
                    img_embeddings = F.normalize(image_outputs[:, 0], p=2, dim=1)
                    all_image_embeddings.append(img_embeddings.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.vision_model(**inputs).last_hidden_state
                    img_embeddings = F.normalize(image_outputs[:, 0], p=2, dim=1)
                    all_image_embeddings.append(img_embeddings.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    def calculate_probs(self, text_embeddings, image_embeddings):
        # already normalized in the encoding functions
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)

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


nomic_embed_vision_v1_5 = ModelMeta(
    loader=partial(
        NomicVisionModelWrapper,
        vision_model_name="nomic-ai/nomic-embed-vision-v1.5",
        text_model_name="nomic-ai/nomic-embed-text-v1.5",
    ),
    name="nomic-ai/nomic-embed-vision-v1.5",
    languages=["eng_Latn"],
    revision="af2246fffdab78d8458418480e4886a8e48b70a7",
    release_date="2024-06-08",
    modalities=["image", "text"],
    n_parameters=92_900_000,
    memory_usage_mb=355,
    max_tokens=2048,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/contrastors",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # https://arxiv.org/pdf/2406.18587
        # DFN-2B
    },
)
