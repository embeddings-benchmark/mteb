from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Blip2TextModelWithProjection,
    Blip2VisionModelWithProjection,
)

from mteb.model_meta import ModelMeta


def blip2_loader(**kwargs):
    class BLIP2ModelWrapper:
        def __init__(
            self,
            model_name: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            self.model_name = model_name
            self.device = device
            # self.processor = Blip2Processor.from_pretrained(model_name)
            self.vision_model = Blip2VisionModelWithProjection.from_pretrained(
                model_name
            ).to(device)
            self.language_model = Blip2TextModelWithProjection.from_pretrained(
                model_name
            ).to(device)
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
                    text_tokens = self.processor(
                        text=batch_texts,
                        padding=True,
                        truncation=True,
                        # max_length=self.language_model.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.language_model.forward(**text_tokens)[
                        "text_embeds"
                    ][:, 0, :]
                    # text_outputs = normalize(self.model.text_proj(text_outputs))
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
                        image_outputs = self.vision_model.forward(
                            inputs["pixel_values"].to(self.device)
                        )
                        image_outputs = image_outputs[0][:, 0, :]
                        # image_outputs = normalize(self.model.vision_proj(image_outputs), dim=-1)
                        all_image_embeddings.append(image_outputs.cpu())
            else:
                with torch.no_grad():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        inputs = self.processor(
                            images=batch_images, return_tensors="pt", padding=True
                        )["pixel_values"].to(self.device)
                        image_outputs = self.vision_model.forward(inputs)
                        image_outputs = image_outputs[0][:, 0, :]
                        # image_outputs = normalize(self.model.vision_proj(image_outputs), dim=-1)
                        all_image_embeddings.append(image_outputs.cpu())

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

        def calculate_probs(self, text_embeddings, image_embeddings):
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
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
                    raise ValueError(
                        f"fusion mode {fusion_mode} hasn't been implemented"
                    )
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

    return BLIP2ModelWrapper(**kwargs)


"""blip2_opt_2_7b = ModelMeta(
    loader=partial(
        blip2_loader,
        model_name="Salesforce/blip2-opt-2.7b",
    ),
    name="Salesforce/blip2-opt-2.7b",
    languages=["eng_Latn"],
    open_source=True,
    revision="ec3c346f4aff99452053bae7abd1753401e6f28a",
    release_date="2024-11-23",
)

blip2_opt_6_7b_coco = ModelMeta(
    loader=partial(
        blip2_loader,
        model_name="Salesforce/blip2-opt-6.7b-coco",
    ),
    name="Salesforce/blip2-opt-6.7b-coco",
    languages=["eng_Latn"],
    open_source=True,
    revision="0d580de59320a25a4d2c386387bcef310d5f286e",
    release_date="2024-03-31",
)"""

blip2_itm_vit_g = ModelMeta(
    loader=partial(
        blip2_loader,
        model_name="Salesforce/blip2-itm-vit-g",
    ),
    name="Salesforce/blip2-itm-vit-g",
    languages=["eng_Latn"],
    open_source=True,
    revision="6e2c730e768eb499b886c73935a539ca2c197290",
    release_date="2024-11-04",
)


blip2_itm_vit_g_coco = ModelMeta(
    loader=partial(
        blip2_loader,
        model_name="Salesforce/blip2-itm-vit-g-coco",
    ),
    name="Salesforce/blip2-itm-vit-g-coco",
    languages=["eng_Latn"],
    open_source=True,
    revision="b1c997706e13245bbd2af5b2717239488d849f07",
    release_date="2024-11-04",
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(blip2_itm_vit_g.name, blip2_itm_vit_g.revision, device="cuda")

    cat_img = Image.open("cat.jpg")
    image_cat_emb = mdl.get_image_embeddings([cat_img])

    emb = mdl.get_text_embeddings(["Hello, world!"])
    emb2 = mdl.get_text_embeddings(["Hello there, world!"])
    emb3 = mdl.get_text_embeddings(["Goodbye, person!"])

    sim = torch.nn.functional.cosine_similarity(emb, emb2)
    print(sim)

    sim = torch.nn.functional.cosine_similarity(emb, emb3)
    print(sim)

    cat_text = "An image of a cat"

    multi_cat_emb = mdl.get_fused_embeddings(["A photo of an animal"], [cat_img])
    multi_conflicting_emb = mdl.get_fused_embeddings(["A photo of a dog"], [cat_img])
    text_cat_emb = mdl.get_text_embeddings(["An photo of a cat"])
    text_dog_emb = mdl.get_text_embeddings(["An image of a dog"])

    print(multi_cat_emb.shape)

    sim1 = torch.nn.functional.cosine_similarity(image_cat_emb, text_cat_emb)
    sim2 = torch.nn.functional.cosine_similarity(image_cat_emb, text_dog_emb)
    sim3 = torch.nn.functional.cosine_similarity(multi_cat_emb, text_cat_emb)
    sim4 = torch.nn.functional.cosine_similarity(multi_cat_emb, text_dog_emb)
    sim5 = torch.nn.functional.cosine_similarity(multi_conflicting_emb, text_cat_emb)

    print(sim1, sim2)

    print(sim3, sim4, sim5)
