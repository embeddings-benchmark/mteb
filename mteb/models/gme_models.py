from __future__ import annotations

import logging
import math
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class GMEWrapper(Wrapper):
    """
    GME implementation.

    source: https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_image_tokens=256,
        max_image_tokens=1280,
        max_length=1800,
        **kwargs,
    ) -> None:
        """Wrapper for GME models.

        Args:
            model_name: The GME model to load from HuggingFace Hub.
            **kwargs: Additional arguments to pass to the wrapper.
        """
        self.model_name = model_name
        self.base = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch.float16, **kwargs
        )
        self.base.eval()
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )
        self.processor.tokenizer.padding_side = "right"
        self.defualt_instruction = "You are a helpful assistant."
        self.sep = " "

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        # pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: torch.LongTensor | None = None,
        # video_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.base.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.base.visual.get_dtype())
                image_embeds = self.base.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            # if pixel_values_videos is not None:
            #     pixel_values_videos = pixel_values_videos.type(self.base.visual.get_dtype())
            #     video_embeds = self.base.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
            #     video_mask = input_ids == self.base.config.video_token_id
            #     inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.base.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]  # TODO
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            embeddings = outputs.last_hidden_state[
                torch.arange(batch_size, device=outputs.last_hidden_state.device),
                sequence_lengths,
            ]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def embed(
        self,
        texts: list[str],
        images: list,
        is_query=True,
        instruction=None,
        **kwargs,
    ):
        self.base.to(self.device)
        # Inputs must be batched
        input_texts, input_images = list(), list()
        for t, i in zip(texts, images):
            if not is_query or instruction is None:
                instruction = self.defualt_instruction
            input_str = ""
            if i is None:
                input_images = None  # All examples in the same batch are consistent
            else:
                input_str += "<|vision_start|><|image_pad|><|vision_end|>"
                input_images.append(i)
            if t is not None:
                input_str += t
            msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
            input_texts.append(msg)

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # TODO
        with torch.no_grad():
            embeddings = self.forward(**inputs)
        return embeddings

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list | DataLoader = None,
        **kwargs,
    ):
        if isinstance(images, DataLoader):
            image_loader = images
            batch_size = image_loader.batch_size
            image_loader.dataset.transform = None
        else:
            batch_size = kwargs.pop("batch_size", 32)
            if images is None:
                image_loader = None
            else:
                image_loader = DataLoader(
                    images,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda batch: batch,
                    num_workers=min(math.floor(os.cpu_count() / 2), 8),
                )

        if texts is None:
            assert image_loader is not None
            n_batch = len(image_loader)
        else:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
            image_loader = image_loader or [None] * n_batch

        all_embeddings = list()
        none_batch = [None] * batch_size
        for n, img_batch in zip(
            range(0, n_batch * batch_size, batch_size), image_loader
        ):
            text_batch = none_batch if texts is None else texts[n : n + batch_size]
            img_batch = none_batch if img_batch is None else img_batch
            embeddings = self.embed(texts=text_batch, images=img_batch, **kwargs)
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def encode(self, sentences: list[str], *, prompt_name=None, **kwargs):
        return self.get_fused_embeddings(
            texts=sentences, prompt_name=prompt_name, **kwargs
        )


gme_qwen2_vl_2b_instruct = ModelMeta(
    loader=partial(
        GMEWrapper,
        model_name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    ),
    name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="cfeb66885b598de483cc04eb08c7d9da534d7afe",
    release_date="2024-12-21",
    n_parameters=int(2.21 * 1e9),
    max_tokens=32768,
    embed_dim=1536,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        # Only annotating text data for now
        # source: https://arxiv.org/pdf/2412.16855
        "MSMARCO": ["train"],
        "MSMARCO.v2": ["train"],
    },
    public_training_code=None,
    public_training_data=None,
)

gme_qwen2_vl_7b_instruct = ModelMeta(
    loader=partial(
        GMEWrapper,
        model_name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    ),
    name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="d42eca5a540526cfa982a349724b24b25c12a95e",
    release_date="2024-12-21",
    n_parameters=int(8.29 * 1e9),
    max_tokens=32768,
    embed_dim=3584,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        # Only annotating text data for now
        # source: https://arxiv.org/pdf/2412.16855
        "MSMARCO": ["train"],
        "MSMARCO.v2": ["train"],
    },
    public_training_code=None,
    public_training_data=None,
)
