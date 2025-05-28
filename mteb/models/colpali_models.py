from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
)

logger = logging.getLogger(__name__)


class ColPaliEngineWrapper:
    """Base wrapper for `colpali_engine` models. Adapted from https://github.com/illuin-tech/colpali/tree/bebcdd6715dba42624acd8d7f7222a16a5daf848/colpali_engine/models"""

    def __init__(
        self,
        model_name: str,
        model_class: type,
        processor_class: type,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.mdl = model_class.from_pretrained(
            model_name, revision=revision, device_map=self.device, **kwargs
        )
        self.mdl.eval()

        # Load processor
        self.processor = processor_class.from_pretrained(model_name)

    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
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


class ColPaliWrapper(ColPaliEngineWrapper):
    """Wrapper for ColPali models."""

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColPali, ColPaliProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColPali,
            processor_class=ColPaliProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )


COLPALI_TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
}

colpali_v1_1 = ModelMeta(
    loader=partial(
        ColPaliWrapper,
        model_name="vidore/colpali-v1.1",
        torch_dtype=torch.float16,
    ),
    name="vidore/colpali-v1.1",
    languages=["eng-Latn"],
    revision="a0f15e3bcf97110e7ac1bb4be4bcd30eeb31992a",
    release_date="2024-08-21",
    modalities=["image", "text"],
    n_parameters=2_920_000_000,
    memory_usage_mb=4700,
    max_tokens=16384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colpali-v1.1",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

colpali_v1_2 = ModelMeta(
    loader=partial(
        ColPaliWrapper,
        model_name="vidore/colpali-v1.2",
        torch_dtype=torch.float16,
    ),
    name="vidore/colpali-v1.2",
    languages=["eng-Latn"],
    revision="6b89bc63c16809af4d111bfe412e2ac6bc3c9451",
    release_date="2024-08-26",
    modalities=["image", "text"],
    n_parameters=2_920_000_000,
    memory_usage_mb=4700,
    max_tokens=16384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colpali-v1.2",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

colpali_v1_3 = ModelMeta(
    loader=partial(
        ColPaliWrapper,
        model_name="vidore/colpali-v1.3",
        torch_dtype=torch.float16,
    ),
    name="vidore/colpali-v1.3",
    languages=["eng-Latn"],
    revision="1b5c8929330df1a66de441a9b5409a878f0de5b0",
    release_date="2024-11-01",
    modalities=["image", "text"],
    n_parameters=2_920_000_000,
    memory_usage_mb=4700,
    max_tokens=16384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/vidore/colpali-v1.3",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)
