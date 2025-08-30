from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
)

logger = logging.getLogger(__name__)


class GraniteVisionEmbeddingWrapper:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Load model
        self.mdl = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            device_map=self.device,
            trust_remote_code=True,
            **kwargs,
        )

        self.mdl.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, revision=revision
        )

    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 16,
        **kwargs,
    ):
        import torchvision.transforms.functional as F

        all_embeds = []

        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        with torch.no_grad():
            for batch in iterator:
                # batch may be list of tensors or PIL
                imgs = [
                    F.to_pil_image(b.to("cpu")) if not isinstance(b, Image.Image) else b
                    for b in batch
                ]
                inputs = self.processor.process_images(imgs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_text_embeddings(
        self,
        texts,
        batch_size: int = 32,
        **kwargs,
    ):
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.processor.process_queries(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

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
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def calculate_probs(self, text_embeddings, image_embeddings):
        scores = self.similarity(text_embeddings, image_embeddings)
        return (scores * 100).softmax(dim=-1)

    def similarity(self, a, b):
        return self.processor.score_multi_vector(a, b)


granite_vision_embedding = ModelMeta(
    loader=partial(
        GraniteVisionEmbeddingWrapper,
        model_name="ibm-granite/granite-vision-3.3-2b-embedding",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
        revision="cee615db64d89d1552a4ee39c50f25c0fc5c66ca",
    ),
    name="ibm-granite/granite-vision-3.3-2b-embedding",
    languages=["eng-Latn"],
    revision="cee615db64d89d1552a4ee39c50f25c0fc5c66ca",
    release_date="2025-06-11",
    modalities=["image", "text"],
    n_parameters=2_980_000_000,
    memory_usage_mb=11351,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=None,  # proprietary, not public
)
