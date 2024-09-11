from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.nn.functional import normalize
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForImageTextRetrieval, BlipProcessor

from mteb.model_meta import ModelMeta


class BLIP2ModelWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)

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
                # different to CLIPModelWrapper: text_encoder instead of get_text_features and apply projection and normalization
                text_outputs = self.model.text_encoder(**inputs)
                text_outputs = text_outputs[0]
                text_outputs = normalize(self.model.text_proj(text_outputs[:,0,:]), dim=-1)
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
                    image_outputs = self.model.vision_model(**inputs)
                    image_outputs = image_outputs[0]
                    image_outputs = normalize(self.model.vision_proj(image_outputs[:,0,:]), dim=-1)
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
                    image_outputs = self.model.vision_model(**inputs)
                    image_outputs = image_outputs[0]
                    image_outputs = normalize(self.model.vision_proj(image_outputs[:,0,:]), dim=-1)
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

Salesforce/blip2-opt-2.7b
Image-to-Text • Updated Mar 22 •
588k •
296
Salesforce/blip2-flan-t5-xxl
Image-to-Text • Updated Mar 29 •
9.23k •
84
Salesforce/blip2-opt-6.7b-coco
Image-to-Text • Updated Mar 31 •
1.51k •
28
Salesforce/blip2-opt-6.7b
Image-to-Text • Updated Mar 27 •
4.93k •
71
Salesforce/blip2-flan-t5-xl
Image-to-Text • Updated Dec 13, 2023 •
95.9k •
56
"""
# in descending order of usage (downloads from huggingface)

blip2_opt_2_7b = ModelMeta(
    loader=partial(
        BLIP2ModelWrapper,
        model_name="Salesforce/blip2-opt-2.7b",
    ),
    name="Salesforce/blip2-opt-2.7b",
    languages=["eng_Latn"],
    open_source=True,
    revision="51572668da0eb669e01a189dc22abe6088589a24",
    release_date="2024-03-22",
)

blip2_flan_t5_xxl = ModelMeta(
    loader=partial(
        BLIP2ModelWrapper,
        model_name="Salesforce/blip2-flan-t5-xxl",
    ),
    name="Salesforce/blip2-flan-t5-xxl",
    languages=["eng_Latn"],
    open_source=True,
    revision="43206cbc865b9d5b3dd7d080e5d94b4143ca8e74",
    release_date="2024-03-29",
)

blip2_opt_6_7b_coco = ModelMeta(
    loader=partial(
        BLIP2ModelWrapper,
        model_name="Salesforce/blip2-opt-6.7b-coco",
    ),
    name="Salesforce/blip2-opt-6.7b-coco",
    languages=["eng_Latn"],
    open_source=True,
    revision="0d580de59320a25a4d2c386387bcef310d5f286e",
    release_date="2024-03-31",
)

blip2_opt_6_7b = ModelMeta(
    loader=partial(
        BLIP2ModelWrapper,
        model_name="Salesforce/blip2-opt-6.7b",
    ),
    name="Salesforce/blip2-opt-6.7b",
    languages=["eng_Latn"],
    open_source=True,
    revision="1d33d60155fd1323b97556e0f1dd5148a9749f5b",
    release_date="2024-03-27",
)

blip2_flan_t5_xl = ModelMeta(
    loader=partial(
        BLIP2ModelWrapper,
        model_name="Salesforce/blip2-flan-t5-xl",
    ),
    name="Salesforce/blip2-flan-t5-xl",
    languages=["eng_Latn"],
    open_source=True,
    revision="e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77",
    release_date="2023-12-13",
)

if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(
        blip2_opt_2_7b.name, blip2_opt_2_7b.revision
    )
    emb = mdl.get_text_embeddings(["Hello, world!"])
    emb2 = mdl.get_text_embeddings(["Hello there, world!"])
    emb3 = mdl.get_text_embeddings(["Goodbye, person!"])
    
    sim = torch.nn.functional.cosine_similarity(emb, emb2)
    print(sim)

    sim = torch.nn.functional.cosine_similarity(emb, emb3)
    print(sim)
    
