from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)

logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


class VLM2VecWrapper:
    """Adapted from https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/model.py"""

    def __init__(
        self,
        model_name: str = "TIGER-Lab/VLM2Vec-LoRA",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        requires_image_dependencies()
        if suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        ):
            import flash_attn  # noqa

        requires_package(self, "peft", model_name, "pip install 'mteb[peft]'")
        from peft import LoraConfig, PeftModel  # noqa

        self.pooling = "last"
        self.normalize = True
        self.temperature = 1.0
        self.hidden_size = 4096
        self.device = device

        # Loading the base model
        base_model_name = "microsoft/Phi-3.5-vision-instruct"
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        config.use_cache = False
        config.padding_side = "right"

        checkpoint_path = model_name if model_name else base_model_name
        base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model.padding_side = "right"

        # Building the model on top of the base
        if "LoRA" in model_name:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(
                base_model, checkpoint_path, config=lora_config
            )
            merged_model = lora_model.merge_and_unload()
            model = merged_model.to(torch.bfloat16)  # propagate dtype.
        else:
            model = base_model.to(torch.bfloat16)

        model.eval()
        model.to(device)
        self.mdl = model

        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            num_crops=4,
        )

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        return self.get_text_embeddings(texts=sentences)

    def encode_input(self, input):
        hidden_states = self.mdl(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input["attention_mask"])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == "last":
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    # reference: https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/collator.py
    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        text = "<|image_1|> Represent the given image."
        all_image_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    input_ids, pixel_values, image_sizes = [], [], []
                    for b in batch:
                        inputs = self.processor(
                            text,
                            [F.to_pil_image(b.to("cpu"))],
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_sizes.append(inputs["image_sizes"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_sizes = torch.cat(image_sizes, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_sizes": image_sizes,
                    }

                    image_outputs = self.encode_input(inputs)
                    all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    input_ids, pixel_values, image_sizes = [], [], []
                    for b in batch_images:
                        inputs = self.processor(
                            text,
                            [b],
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_sizes.append(inputs["image_sizes"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_sizes = torch.cat(image_sizes, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_sizes": image_sizes,
                    }

                    image_outputs = self.encode_input(inputs)
                    all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

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
                input_ids = []
                batch_texts = texts[i : i + batch_size]
                for text in batch_texts:
                    inputs = self.processor(
                        text,
                        None,
                        return_tensors="pt",
                        max_length=256,
                        truncation=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))

                input_ids = torch._C._nn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                ).squeeze(2)
                attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                text_outputs = self.encode_input(inputs)
                all_text_embeddings.append(text_outputs.cpu().to(torch.float32))

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

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
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None
        kwargs.update(
            task_name=task_name, prompt_type=prompt_type, batch_size=batch_size
        )

        if texts is not None and images is None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)
            return text_embeddings

        if images is not None and texts is None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)
            return image_embeddings

        # text_embeddings is not None and image_embeddings is not None
        texts = iter(texts)
        all_fused_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in images:
                    input_ids, pixel_values, image_sizes = [], [], []
                    for b in batch:
                        text = next(texts)
                        inputs = self.processor(
                            f"<|image_1|> Represent the given image with the following question: {text}",
                            [F.to_pil_image(b.to("cpu"))],
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_sizes.append(inputs["image_sizes"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_sizes = torch.cat(image_sizes, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_sizes": image_sizes,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    input_ids, pixel_values, image_sizes = [], [], []
                    for b in batch_images:
                        text = next(texts)
                        inputs = self.processor(
                            f"<|image_1|> Represent the given image with the following question: {text}",
                            [b],
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"])
                        image_sizes.append(inputs["image_sizes"])

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_sizes = torch.cat(image_sizes, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_sizes": image_sizes,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))

        fused_embeddings = torch.cat(all_fused_embeddings, dim=0)
        return fused_embeddings


vlm2vec_training_datasets = {
    # MMEB-train
}

vlm2vec_lora = ModelMeta(
    loader=partial(
        VLM2VecWrapper,
        model_name="TIGER-Lab/VLM2Vec-LoRA",
    ),
    name="TIGER-Lab/VLM2Vec-LoRA",
    languages=["eng_Latn"],
    revision="7403b6327958071c1e33c822c7453adadccc7298",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    framework=["PyTorch"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-LoRA",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
)

vlm2vec_full = ModelMeta(
    loader=partial(
        VLM2VecWrapper,
        model_name="TIGER-Lab/VLM2Vec-Full",
    ),
    name="TIGER-Lab/VLM2Vec-Full",
    languages=["eng_Latn"],
    revision="e9afa98002097ac2471827ba23ea1f2ddd229480",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=4_150_000_000,
    memory_usage_mb=7909,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    framework=["PyTorch"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
)
