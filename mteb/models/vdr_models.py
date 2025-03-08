from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class VDRWrapper(Wrapper):
    def __init__(
        self,
        model_name: str = "llamaindex/vdr-2b-multi-v1",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        """Wrapper for VDR multimodal model that supports both image and text.

        Args:
            model_name: Path to model or model name
            device: Device to use (cuda, cpu)
            **kwargs: Additional arguments to pass to the model
        """
        super().__init__()
        self.model_name = model_name
        self.max_pixels = kwargs.get("max_pixels", 768 * 28 * 28)
        self.min_pixels = kwargs.get("min_pixels", 1 * 28 * 28)

        # Load processor with image size constraints
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        use_flash_attention = (
            "cuda" in device
            and kwargs.get("attn_implementation", "flash_attention_2")
            == "flash_attention_2"
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager"
            if not use_flash_attention
            else "flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Set model configuration
        self.model.eval()
        self.model.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        self.device = device
        self.embed_dim = kwargs.get("embed_dim")

        # Define standard prompts for text and queries
        self.document_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
        self.query_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"
        self.text_only_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|endoftext|>"

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Returns embeddings for text inputs."""
        return self.get_text_embeddings(sentences, **kwargs)

    def get_text_embeddings(
        self, texts: list[str], batch_size: int = 8, **kwargs: Any
    ) -> torch.Tensor:
        """Get embeddings for text inputs (without images).

        Args:
            texts: list of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments to pass to the model

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        dimension = kwargs.get("dimension", self.embed_dim)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            formatted_texts = [self.text_only_prompt % text for text in batch_texts]

            inputs = self.processor(
                text=formatted_texts,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            seq_length = inputs["input_ids"].shape[1]
            cache_position = torch.arange(0, seq_length).to(self.device)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )
            with torch.no_grad():
                output = self.model(
                    **inputs, return_dict=True, output_hidden_states=True
                )

            embeddings = output.hidden_states[-1][:, -1]
            norm_embeddings = torch.nn.functional.normalize(
                embeddings[:, :dimension], p=2, dim=-1
            )

            all_embeddings.append(norm_embeddings.detach().cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Get embeddings for image inputs.

        Args:
            images: list of PIL images or DataLoader of images
            batch_size: Batch size for processing
            **kwargs: Additional arguments to pass to the model

        Returns:
            Numpy array of embeddings
        """
        dimension = kwargs.get("dimension", self.embed_dim)
        all_embeddings = []

        if isinstance(images, DataLoader):
            batches = images
        else:
            batches = [
                images[i : i + batch_size] for i in range(0, len(images), batch_size)
            ]

        for batch in batches:
            batch_images = batch["image"] if isinstance(images, DataLoader) else batch
            inputs = self.processor(
                text=[self.document_prompt] * len(batch_images),
                images=batch_images,
                padding="longest",
                truncation=True,
                max_length=kwargs.get("max_length", 512),
                return_tensors="pt",
            ).to(self.device)

            seq_length = inputs["input_ids"].shape[1]
            cache_position = torch.arange(0, seq_length).to(self.device)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )

            with torch.no_grad():
                output = self.model(
                    **inputs, return_dict=True, output_hidden_states=True
                )

            embeddings = output.hidden_states[-1][:, -1]
            norm_embeddings = torch.nn.functional.normalize(
                embeddings[:, :dimension], p=2, dim=-1
            )
            all_embeddings.append(norm_embeddings.detach().cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def get_fused_embeddings(
        self, texts: list[str], images: list[Image.Image], **kwargs: Any
    ) -> torch.Tensor:
        """Get embeddings for text and image pairs.

        Args:
            texts: list of text strings to use as queries
            images: list of PIL images
            **kwargs: Additional arguments to pass to the model

        Returns:
            Numpy array of fused embeddings
        """
        assert len(texts) == len(images), "Number of texts and images must be equal"

        all_embeddings = []
        batch_size = kwargs.get("batch_size", 8)
        dimension = kwargs.get("dimension", self.embed_dim)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_images = images[i : i + batch_size]

            formatted_texts = [self.query_prompt % text for text in batch_texts]

            inputs = self.processor(
                text=formatted_texts,
                images=batch_images,
                padding="longest",
                truncation=True,
                max_length=kwargs.get("max_length", 512),
                return_tensors="pt",
            ).to(self.device)

            seq_length = inputs["input_ids"].shape[1]
            cache_position = torch.arange(0, seq_length).to(self.device)
            inputs = self.model.prepare_inputs_for_generation(
                **inputs, cache_position=cache_position, use_cache=False
            )

            with torch.no_grad():
                output = self.model(
                    **inputs, return_dict=True, output_hidden_states=True
                )

            embeddings = output.hidden_states[-1][:, -1]
            norm_embeddings = torch.nn.functional.normalize(
                embeddings[:, :dimension], p=2, dim=-1
            )

            all_embeddings.append(norm_embeddings.detach().cpu())

        return torch.cat(all_embeddings, dim=0).numpy()


languages = [
    "eng_Latn",
    "ita_Latn",
    "fra_Latn",
    "deu_Latn",
    "spa_Latn",
]
vdr_training_sets = {
    # llamaindex/vdr-multilingual-train
}
vdr_2b_multi_v1 = ModelMeta(
    loader=partial(
        VDRWrapper,
        model_name="llamaindex/vdr-2b-multi-v1",
    ),
    name="llamaindex/vdr-2b-multi-v1",
    languages=languages,
    open_weights=True,
    revision="2c4e54c8db4071cc61fc3c62f4490124e40c37db",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=2_000_000_000,
    memory_usage_mb=4213,
    max_tokens=32768,
    embed_dim=1536,
    license="apache-2.0",
    reference="https://huggingface.co/llamaindex/vdr-2b-multi-v1",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/llamaindex/vdr-multilingual-train",
    training_datasets=vdr_training_sets,
)
