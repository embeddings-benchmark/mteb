from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

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
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def preprocess(
        self,
        texts: list[str],
        images: list[Image.Image],
    ):
        return self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

    def get_text_embeddings(self, texts: list[str], batch_size: int = 32):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.get_text_features(**inputs)
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
                    inputs = self.processor(
                        images=batch, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.get_image_features(**inputs)
                    all_image_embeddings.append(image_outputs.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(
                        images=batch_images, return_tensors="pt", padding=True
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

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        fusion_mode="sum",
        batch_size: int = 32,
    ):
        # TODO: find out if BLIP has a prescribed way of fusing text and image embeddings
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, batch_size)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, batch_size)

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


"""
TODO: implement all model variants

Salesforce/blip-image-captioning-large
Image-to-Text • Updated Dec 7, 2023 •
1.16M •
•
1.04k
Salesforce/blip-image-captioning-base
Image-to-Text • Updated Aug 1, 2023 •
857k •
•
475
Salesforce/blip-vqa-base
Visual Question Answering • Updated Dec 7, 2023 •
168k •
119
Salesforce/blip-vqa-capfilt-large
Visual Question Answering • Updated Jan 22 •
90.6k •
44
Salesforce/blip-itm-base-coco
Updated Aug 1, 2023 •
12.8k •
16
Salesforce/blip-itm-large-coco
Updated Aug 1, 2023 •
9.9k
Salesforce/blip-itm-base-flickr
Updated Aug 1, 2023 •
65
Salesforce/blip-itm-large-flickr
Updated Aug 1, 2023 •
459 •
2
"""
# in descending order of usage (downloads from huggingface)
blip_image_captioning_large = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-image-captioning-large",
    ),
    name="Salesforce/blip-image-captioning-large",
    languages=["eng_Latn"],
    open_source=True,
    revision="2227ac38c9f16105cb0412e7cab4759978a8fd90",
    release_date="2023-12-07",
)

blip_image_captioning_base = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-image-captioning-base",
    ),
    name="Salesforce/blip-image-captioning-base",
    languages=["eng_Latn"],
    open_source=True,
    revision="89b09ea1789f7addf2f6d6f0dfc4ce10ab58ef84",
    release_date="2023-08-01",
)


blip_vqa_base = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-vqa-base",
    ),
    name="Salesforce/blip-vqa-base",
    languages=["eng_Latn"],
    open_source=True,
    revision="c7df8e7cd7aa2ee9af18f56e2b29e59a92651b64",
    release_date="2023-12-07",
)

blip_vqa_capfilt_large = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-vqa-capfilt-large",
    ),
    name="Salesforce/blip-vqa-capfilt-large",
    languages=["eng_Latn"],
    open_source=True,
    revision="e53f95265aeab69013fabb5380500ab984adbbb4",
    release_date="2023-01-22",
)

blip_itm_base_coco = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-base-coco",
    ),
    name="Salesforce/blip-itm-base-coco",
    languages=["eng_Latn"],
    open_source=True,
    revision="7eaa90c11850c0b17fc38c6a11e7d88bd6ac231f",
    release_date="2023-08-01",
)

blip_itm_large_coco = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-large-coco",
    ),
    name="Salesforce/blip-itm-large-coco",
    languages=["eng_Latn"],
    open_source=True,
    revision="fef05cafc05298067cbbca00b125749394a77a6f",
    release_date="2023-08-01",
)

blip_itm_base_flickr = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-base-flickr",
    ),
    name="Salesforce/blip-itm-base-flickr",
    languages=["eng_Latn"],
    open_source=True,
    revision="1de29e660d91ae1786c1876212ea805a22eab251",
    release_date="2023-08-01",
)

blip_itm_large_flickr = ModelMeta(
    loader=partial(
        BLIPModelWrapper,
        model_name="Salesforce/blip-itm-large-flickr",
    ),
    name="Salesforce/blip-itm-large-flickr",
    languages=["eng_Latn"],
    open_source=True,
    revision="bda12e6506758f54261b5ab174b2c55a3ba143fb",
    release_date="2023-08-01",
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(
        blip_image_captioning_base.name, blip_image_captioning_base.revision
    )
    emb = mdl.get_text_embeddings(["Hello, world!"])
    print(emb.shape)
