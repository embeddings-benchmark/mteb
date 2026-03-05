from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
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
  url={https://nomic.ai/blog/posts/nomic-embed-multimodal}
}"""

# https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source
TRAINING_DATA: set[str] = set()


logger = logging.getLogger(__name__)


class BiQwen2_5Wrapper(AbsEncoder):  # noqa: N801
    """Wrapper for BiQwen2_5 dense (single-vector) embedding model."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.mdl = BiQwen2_5.from_pretrained(
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.mdl.eval()

        # Load processor
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

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                # batch may be list of tensors or PIL
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
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
            for batch in tqdm(texts, desc="Encoding texts"):
                inputs = self.processor.process_queries(batch["text"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def calculate_probs(self, text_embeddings, image_embeddings):
        scores = self.similarity(text_embeddings, image_embeddings).T
        return scores.softmax(dim=-1)

    def similarity(self, a, b):
        return self.processor.score(a, b, device=self.device)


nomic_embed_multimodal_3b = ModelMeta(
    loader=BiQwen2_5Wrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="nomic-ai/nomic-embed-multimodal-3b",
    model_type=["dense"],
    languages=NOMIC_LANGUAGES,
    revision="main",  # Will need to be updated with actual revision
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=6200,  # Estimated based on 3B vs 7B scaling
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
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="nomic-ai/nomic-embed-multimodal-7b",
    model_type=["dense"],
    languages=NOMIC_LANGUAGES,
    revision="1291f1b6ca07061b0329df9d5713c09b294be576",
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    n_embedding_parameters=None,
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
