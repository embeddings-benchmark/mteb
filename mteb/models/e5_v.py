from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import transformers
from packaging import version
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from mteb.encoder_interface import BatchedInput, PromptType
from mteb.model_meta import ModelMeta

E5_V_TRANSFORMERS_VERSION = (
    "4.44.2"  # Issue 1647: Only works with transformers==4.44.2.
)


class E5VWrapper:
    def __init__(
        self,
        model_name: str,
        composed_prompt=None,
        **kwargs: Any,
    ):
        if version.parse(transformers.__version__) > version.parse(
            E5_V_TRANSFORMERS_VERSION
        ):
            raise ImportError(
                f"This wrapper only works with transformers=={E5_V_TRANSFORMERS_VERSION}"
            )

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

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                text_inputs = self.processor(
                    [
                        self.text_prompt.replace("<sent>", text)
                        for text in batch["text"]
                    ],
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                text_outputs = self.model(
                    **text_inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                all_text_embeddings.append(text_outputs.cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                img_inputs = self.processor(
                    [self.img_prompt] * len(batch["image"]),
                    batch["image"],
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

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        fusion_mode: Literal["sum"] = "sum",
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        if "image" in inputs.dataset.features and "text" in inputs.dataset.features:
            all_fused_embeddings = []

            with torch.no_grad():
                for batch in tqdm(
                    inputs, disable=not show_progress_bar, desc="Fused Encoding"
                ):
                    prompts = [
                        self.composed_prompt.format(text) for text in batch["text"]
                    ]
                    inputs = self.processor(
                        prompts, batch["image"], return_tensors="pt", padding=True
                    ).to("cuda")
                    outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    ).hidden_states[-1][:, -1, :]
                    all_fused_embeddings.append(outputs.cpu())
            return torch.cat(all_fused_embeddings, dim=0)
        elif "text" in inputs.dataset.features:
            return self.get_text_embeddings(inputs, **kwargs)
        elif "image" in inputs.dataset.features:
            return self.get_image_embeddings(inputs, **kwargs)


e5_v = ModelMeta(
    loader=E5VWrapper,
    loader_kwargs=dict(
        device_map="auto",
    ),
    name="royokong/e5-v",
    languages=["eng_Latn"],
    revision="0c1f22679417b3ae925d779442221c40cd1861ab",
    release_date="2024-07-17",
    modalities=["image", "text"],
    n_parameters=8_360_000_000,
    memory_usage_mb=15936,
    max_tokens=8192,
    embed_dim=4096,
    license=None,
    open_weights=True,
    public_training_code="https://github.com/kongds/E5-V",
    public_training_data="https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse",
    framework=["PyTorch"],
    reference="https://huggingface.co/royokong/e5-v",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        # princeton-nlp/datasets-for-simcse
    },
)
