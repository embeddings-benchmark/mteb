from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from mteb.model_meta import ModelMeta


class E5VWrapper:
    def __init__(
        self,
        model_name: str,
        composed_prompt=None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        if "device" in kwargs:
            self.device = kwargs.pop("device")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, **kwargs
        )
        self.model.eval()
        self.template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
        self.text_prompt = self.template.format(
            "<sent>\nSummary above sentence in one word: "
        )
        self.img_prompt = self.template.format(
            "<image>\nSummary above image in one word: "
        )
        if not composed_prompt:
            # default composed embedding, to_do: move it to get_fused_embedding with "prompt_name" like MTEB text ones.
            self.composed_prompt = self.template.format(
                '[INST] <image> Modify this image with "{}" Describe modified image in one word: [/INST]'
            )
        else:
            self.composed_prompt = self.template.format(composed_prompt)

    def get_text_embeddings(self, texts: list[str], batch_size: int = 8):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                text_inputs = self.processor(
                    [self.text_prompt.replace("<sent>", text) for text in batch_texts],
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                text_outputs = self.model(
                    **text_inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                all_text_embeddings.append(text_outputs.cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self, images: list[Image.Image] | DataLoader, batch_size: int = 8
    ):
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            for batch_images in tqdm(images):
                img_inputs = self.processor(
                    [self.img_prompt] * len(batch_images),
                    batch_images,
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                image_outputs = self.model(
                    **img_inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                all_image_embeddings.append(image_outputs.cpu())
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i : i + batch_size]
                img_inputs = self.processor(
                    [self.img_prompt] * len(batch_images),
                    batch_images,
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                image_outputs = self.model(
                    **img_inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                all_image_embeddings.append(image_outputs.cpu())
        return torch.cat(all_image_embeddings, dim=0)

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
        batch_size: int = 8,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        all_fused_embeddings = []

        if texts is not None and images is not None:
            if isinstance(images, DataLoader):
                with torch.no_grad():
                    for index, batch_images in enumerate(tqdm(images)):
                        batch_texts = texts[
                            index * batch_size : (index + 1) * batch_size
                        ]
                        prompts = [
                            self.composed_prompt.format(text) for text in batch_texts
                        ]
                        inputs = self.processor(
                            prompts, batch_images, return_tensors="pt", padding=True
                        ).to("cuda")
                        outputs = self.model(
                            **inputs, output_hidden_states=True, return_dict=True
                        ).hidden_states[-1][:, -1, :]
                        all_fused_embeddings.append(outputs.cpu())
            else:
                if len(texts) != len(images):
                    raise ValueError(
                        "The number of texts and images must have the same length"
                    )
                with torch.no_grad():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_texts = texts[i : i + batch_size]
                        batch_images = images[i : i + batch_size]
                        prompts = [
                            self.composed_prompt.format(text) for text in batch_texts
                        ]
                        inputs = self.processor(
                            prompts, batch_images, return_tensors="pt", padding=True
                        ).to("cuda")
                        outputs = self.model(
                            **inputs, output_hidden_states=True, return_dict=True
                        ).hidden_states[-1][:, -1, :]
                        all_fused_embeddings.append(outputs.cpu())
            return torch.cat(all_fused_embeddings, dim=0)

        elif texts is not None:
            text_embeddings = self.get_text_embeddings(texts, batch_size)
            return text_embeddings
        elif images is not None:
            image_embeddings = self.get_image_embeddings(images, batch_size)
            return image_embeddings


e5_v = ModelMeta(
    loader=partial(
        E5VWrapper,
        model_name="royokong/e5-v",
        torch_dtype=torch.float16,
        device_map="auto",
    ),
    name="royokong/e5-v",
    languages=["eng_Latn"],
    open_source=True,
    revision="0c1f22679417b3ae925d779442221c40cd1861ab",
    release_date="2024-07-17",
)

if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(e5_v.name, e5_v.revision)
    emb = mdl.get_text_embeddings(["Hello, world!"])
