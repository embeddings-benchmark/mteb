from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.colpali_models import COLPALI_TRAINING_DATA
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


NOMIC_LANGUAGES = [
    "deu-Latn",  # German
    "spa-Latn",  # Spanish
    "eng-Latn",  # English
    "fra-Latn",  # French
    "ita-Latn",  # Italian
]

CITATION = """
@misc{nomicembedmultimodal2025,
  title={Nomic Embed Multimodal: Interleaved Text, Image, and Screenshots for Visual Document Retrieval},
  author={Nomic Team},
  year={2025},
  publisher={Nomic AI},
  url={https://www.nomic.ai/news/nomic-embed-multimodal}
}"""

# https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source
TRAINING_DATA = COLPALI_TRAINING_DATA | {"VDRMultilingualRetrieval"}


logger = logging.getLogger(__name__)


class BiQwen2_5Wrapper(AbsEncoder):  # noqa: N801
    """Wrapper for BiQwen2_5 dense (single-vector) embedding model."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        base_revision: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mdl = BiQwen2_5.from_pretrained(
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            revision=base_revision,
            attn_implementation="flash_attention_2",  # With this enabled it goes from 0.57382 to 0.58021 on Vidore2ESGReportsHLRetrieval
            **kwargs,
        )
        self.mdl.eval()

        self.processor = BiQwen2_5_Processor.from_pretrained(model_name)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                inputs = self.processor.process_images(batch["image"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.mdl(**inputs)
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
            for batch in tqdm(texts, desc="Encoding texts"):
                inputs = self.processor.process_queries(batch["text"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.mdl(**inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def similarity(
        self, a, b
    ):  # Using the processing it goes from 0.57382 to 0.57297 on Vidore2ESGReportsHLRetrieval (without flash attention 2)
        return self.processor.score(a, b, device=self.device)


nomic_embed_multimodal_3b = ModelMeta(
    loader=BiQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
        base_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    ),
    name="nomic-ai/nomic-embed-multimodal-3b",
    model_type=["dense"],
    languages=NOMIC_LANGUAGES,
    revision="298930bb768c50b91d2799d6f3b0daf46ea52e70",
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=3_814_490_112,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=6200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=CITATION,
)

nomic_embed_multimodal_7b = ModelMeta(
    loader=BiQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
        base_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
    ),
    name="nomic-ai/nomic-embed-multimodal-7b",
    model_type=["dense"],
    languages=NOMIC_LANGUAGES,
    revision="1291f1b6ca07061b0329df9d5713c09b294be576",
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=7_827_909_632,
    n_embedding_parameters=544997376,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=CITATION,
)
