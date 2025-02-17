from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForImageTextRetrieval, BlipProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


class BLIPModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(
            self.device
        )
        self.processor = BlipProcessor.from_pretrained(model_name)

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
                inputs = self.processor(
                    text=batch_texts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # different to CLIPModelWrapper: text_encoder instead of get_text_features and apply projection and normalization
                text_outputs = self.model.text_encoder(**inputs)
                text_outputs = text_outputs[0]
                text_outputs = normalize(
                    self.model.text_proj(text_outputs[:, 0, :]), dim=-1
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
        **kwargs: Any,
    ):
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    inputs = self.processor(
                        images=batch, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.vision_model(**inputs)
                    image_outputs = image_outputs[0]
                    image_outputs = normalize(
                        self.model.vision_proj(image_outputs[:, 0, :]), dim=-1
                    )
                    all_image_embeddings.append(image_outputs.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(
                        images=batch_images, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.vision_model(**inputs)
                    image_outputs = image_outputs[0]
                    image_outputs = normalize(
                        self.model.vision_proj(image_outputs[:, 0, :]), dim=-1
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


# in descending order of usage (downloads from huggingface)
blip_image_captioning_large = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-image-captioning-large",
    ),
    name="Salesforce/blip-image-captioning-large",
    languages=["eng_Latn"],
    revision="2227ac38c9f16105cb0412e7cab4759978a8fd90",
    release_date="2023-12-07",
    modalities=["image", "text"],
    n_parameters=470_000_000,
    memory_usage_mb=1792,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-image-captioning-large",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # COCO
        # CC3M+CC12M+SBU
        # LAION115M
    },
)

blip_image_captioning_base = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-image-captioning-base",
    ),
    name="Salesforce/blip-image-captioning-base",
    languages=["eng_Latn"],
    revision="89b09ea1789f7addf2f6d6f0dfc4ce10ab58ef84",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-image-captioning-base",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # COCO
        # CC3M+CC12M+SBU
        # LAION115M
    },
)


blip_vqa_base = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-vqa-base",
    ),
    name="Salesforce/blip-vqa-base",
    languages=["eng_Latn"],
    revision="c7df8e7cd7aa2ee9af18f56e2b29e59a92651b64",
    release_date="2023-12-07",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=1467,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-vqa-base",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # CC3M+CC12M+SBU
        # LAION115M
    },
)

blip_vqa_capfilt_large = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-vqa-capfilt-large",
    ),
    name="Salesforce/blip-vqa-capfilt-large",
    languages=["eng_Latn"],
    revision="e53f95265aeab69013fabb5380500ab984adbbb4",
    release_date="2023-01-22",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-vqa-capfilt-large",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # CC3M+CC12M+SBU
        # LAION115M
    },
)

blip_itm_base_coco = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-base-coco",
    ),
    name="Salesforce/blip-itm-base-coco",
    languages=["eng_Latn"],
    revision="7eaa90c11850c0b17fc38c6a11e7d88bd6ac231f",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-base-coco",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # CC3M+CC12M+SBU
        # LAION115M
    },
)

blip_itm_large_coco = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-large-coco",
    ),
    name="Salesforce/blip-itm-large-coco",
    languages=["eng_Latn"],
    revision="fef05cafc05298067cbbca00b125749394a77a6f",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=470_000_000,
    memory_usage_mb=1793,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-large-coco",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # COCO
        # CC3M+CC12M+SBU
        # LAION115M
    },
)

blip_itm_base_flickr = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-base-flickr",
    ),
    name="Salesforce/blip-itm-base-flickr",
    languages=["eng_Latn"],
    revision="1de29e660d91ae1786c1876212ea805a22eab251",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-base-flickr",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # CC3M+CC12M+SBU
        # LAION115M
        # Flickr30k
    },
)

blip_itm_large_flickr = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-large-flickr",
    ),
    name="Salesforce/blip-itm-large-flickr",
    languages=["eng_Latn"],
    revision="bda12e6506758f54261b5ab174b2c55a3ba143fb",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=470_000_000,
    memory_usage_mb=1793,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-large-flickr",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # CC3M+CC12M+SBU
        # LAION115M
    },
)
