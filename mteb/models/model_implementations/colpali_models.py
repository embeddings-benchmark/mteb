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
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class ColPaliEngineWrapper(AbsEncoder):
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
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.mdl.eval()

        # Load processor
        self.processor = processor_class.from_pretrained(model_name)

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
        """Encode inputs using the appropriate modality.

        For ColPali-family visual document retrieval models:
        - Queries are always encoded as text.
        - Documents are encoded as images when available, since the page
          screenshot is the primary representation. When a dataset has both
          "text" and "image" features (e.g. ViDoRe V3 corpus entries with
          OCR text + page screenshots), using images avoids an invalid
          element-wise addition of multi-vector embeddings with different
          sequence lengths.
        """
        from mteb.types import PromptType

        features = inputs.dataset.features
        has_image = "image" in features
        has_text = "text" in features

        if prompt_type == PromptType.query and has_text:
            return self.get_text_embeddings(inputs, **kwargs)
        if has_image:
            return self.get_image_embeddings(inputs, **kwargs)
        elif has_text:
            return self.get_text_embeddings(inputs, **kwargs)
        raise ValueError("No text or image features found in inputs.")

    def encode_input(self, inputs):
        # Clear stale rope_deltas cache to avoid shape mismatches across batches
        if hasattr(self.mdl, "rope_deltas"):
            self.mdl.rope_deltas = None
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
                batch = [
                    self.processor.query_prefix
                    + t
                    + self.processor.query_augmentation_token * 10
                    for t in batch["text"]
                ]
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
        scores = self.similarity(text_embeddings, image_embeddings).T
        return scores.softmax(dim=-1)

    def similarity(self, a, b):
        return self.processor.score(a, b, device=self.device)


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


COLPALI_CITATION = """
@misc{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Omrani, Bilel and Viaud, Gautier and Hudelot, C\'eline and Colombo, Pierre},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}"""

COLPALI_TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
}

colpali_v1_1 = ModelMeta(
    loader=ColPaliWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colpali-v1.1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="a0f15e3bcf97110e7ac1bb4be4bcd30eeb31992a",
    release_date="2024-08-21",
    modalities=["image", "text"],
    n_parameters=2_920_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=4700,
    max_tokens=16384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/vidore/colpali-v1.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

colpali_v1_2 = ModelMeta(
    loader=ColPaliWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colpali-v1.2",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6b89bc63c16809af4d111bfe412e2ac6bc3c9451",
    release_date="2024-08-26",
    modalities=["image", "text"],
    n_parameters=2_920_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=4700,
    max_tokens=16384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/vidore/colpali-v1.2",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

colpali_v1_3 = ModelMeta(
    loader=ColPaliWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colpali-v1.3",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="1b5c8929330df1a66de441a9b5409a878f0de5b0",
    release_date="2024-11-01",
    modalities=["image", "text"],
    n_parameters=2_920_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=4700,
    max_tokens=16384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/vidore/colpali-v1.3",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)
