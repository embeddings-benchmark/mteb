"""
SauerkrautLM Visual Document Retrieval Models - MTEB Integration

This module provides MTEB wrappers for SauerkrautLM ColPali-style models:
- SLM-ColQwen3 (Qwen3-VL backbone)
- SLM-ColLFM2 (LFM2 backbone)
- SLM-ColMinistral3 (Ministral3 backbone)

Based on:
- MTEB ColPali implementation: mteb/models/model_implementations/colpali_models.py
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


# =============================================================================
# Supported Languages
# =============================================================================

SUPPORTED_LANGUAGES = [
    "eng-Latn",  # English
    "deu-Latn",  # German
    "fra-Latn",  # French
    "spa-Latn",  # Spanish
    "ita-Latn",  # Italian
    "por-Latn",  # Portuguese
]


# =============================================================================
# Base Wrapper Class
# =============================================================================

class SLMBaseWrapper(AbsEncoder):
    """
    Base wrapper for SauerkrautLM multi-vector embedding models.
    
    All our models use late interaction (MaxSim) for retrieval scoring.
    """
    
    model_class = None
    processor_class = None
    model_name_prefix = "SLM"

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "sauerkrautlm_colpali", model_name, "pip install sauerkrautlm-colpali"
        )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_processor(model_name, revision, use_flash_attn, **kwargs)

    def _load_model_and_processor(self, model_name, revision, use_flash_attn, **kwargs):
        """Override in subclasses to load specific model/processor."""
        raise NotImplementedError

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
        raise ValueError("No text or image features found in inputs")

    def encode_input(self, inputs):
        """Forward pass through the model."""
        return self.mdl(**inputs)

    def _move_to_device(self, inputs: dict) -> dict:
        """Move all tensor inputs to the model's device."""
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def get_image_embeddings(
        self,
        images: DataLoader,
        batch_size: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        import torchvision.transforms.functional as F

        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                imgs = [
                    F.to_pil_image(b)
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                inputs = self.processor.process_images(imgs)
                inputs = self._move_to_device(inputs)
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_text_embeddings(
        self,
        texts: DataLoader,
        batch_size: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        all_embeds = []
        
        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts"):
                inputs = self.processor.process_queries(batch["text"])
                inputs = self._move_to_device(inputs)
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
        fusion_mode: str = "sum",
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported. "
            "Please use get_text_embeddings or get_image_embeddings."
        )

    def calculate_probs(
        self, 
        text_embeddings: torch.Tensor, 
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.similarity(text_embeddings, image_embeddings).T
        return scores.softmax(dim=-1)

    def similarity(
        self, 
        a: torch.Tensor | list, 
        b: torch.Tensor | list,
    ) -> torch.Tensor:
        return self.processor.score(a, b, device=self.device)


# =============================================================================
# ColQwen3 Wrapper
# =============================================================================

class SLMColQwen3Wrapper(SLMBaseWrapper):
    """Wrapper for SLM-ColQwen3 models (Qwen3-VL backbone)."""

    def _load_model_and_processor(self, model_name, revision, use_flash_attn, **kwargs):
        from sauerkrautlm_colpali.models.qwen3.colqwen3 import ColQwen3, ColQwen3Processor

        self.mdl = ColQwen3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            revision=revision,
            **kwargs,
        )
        # Explicitly move to device
        self.mdl = self.mdl.to(self.device)
        self.mdl.eval()

        self.processor = ColQwen3Processor.from_pretrained(
            model_name,
            revision=revision,
        )
        
        logger.info(f"SLM-ColQwen3 loaded: dim={self.mdl.dim}, device={self.device}")


# =============================================================================
# ColLFM2 Wrapper
# =============================================================================

class SLMColLFM2Wrapper(SLMBaseWrapper):
    """Wrapper for SLM-ColLFM2 models (LFM2 backbone)."""

    def _load_model_and_processor(self, model_name, revision, use_flash_attn, **kwargs):
        from sauerkrautlm_colpali.models.lfm2.collfm2 import ColLFM2, ColLFM2Processor

        self.mdl = ColLFM2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            revision=revision,
            **kwargs,
        )
        # Explicitly move to device
        self.mdl = self.mdl.to(self.device)
        self.mdl.eval()

        self.processor = ColLFM2Processor.from_pretrained(
            model_name,
            revision=revision,
        )
        
        logger.info(f"SLM-ColLFM2 loaded: dim={self.mdl.dim}, device={self.device}")


# =============================================================================
# ColMinistral3 Wrapper
# =============================================================================

class SLMColMinistral3Wrapper(SLMBaseWrapper):
    """Wrapper for SLM-ColMinistral3 models (Ministral3 backbone)."""

    def _load_model_and_processor(self, model_name, revision, use_flash_attn, **kwargs):
        from sauerkrautlm_colpali.models.ministral3.colministral3 import ColMinistral3, ColMinistral3Processor

        # ColMinistral3.__init__ doesn't accept extra kwargs - only pass model_name
        self.mdl = ColMinistral3.from_pretrained(model_name)
        # Explicitly move to device and convert to bfloat16
        self.mdl = self.mdl.to(dtype=torch.bfloat16, device=self.device)
        self.mdl.eval()

        self.processor = ColMinistral3Processor.from_pretrained(model_name)
        
        logger.info(f"SLM-ColMinistral3 loaded: dim={self.mdl.dim}, device={self.device}")


# =============================================================================
# Loader Functions
# =============================================================================

def slm_colqwen3_loader(model_name: str, revision: str | None = None, device: str | None = None, **kwargs) -> SLMColQwen3Wrapper:
    return SLMColQwen3Wrapper(model_name=model_name, revision=revision, device=device, **kwargs)

def slm_collfm2_loader(model_name: str, revision: str | None = None, device: str | None = None, **kwargs) -> SLMColLFM2Wrapper:
    return SLMColLFM2Wrapper(model_name=model_name, revision=revision, device=device, **kwargs)

def slm_colministral3_loader(model_name: str, revision: str | None = None, device: str | None = None, **kwargs) -> SLMColMinistral3Wrapper:
    return SLMColMinistral3Wrapper(model_name=model_name, revision=revision, device=device, **kwargs)


# =============================================================================
# Citations
# =============================================================================

SAUERKRAUTLM_CITATION = """
@misc{sauerkrautlm-colpali-2025,
  title={SauerkrautLM-ColPali: Multi-Vector Vision Retrieval Models},
  author={David Golchinfar},
  organization={VAGO Solutions},
  year={2025},
  url={https://github.com/VAGOsolutions/sauerkrautlm-colpali}
}
"""

COLPALI_CITATION = """
@misc{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Omrani, Bilel and Viaud, Gautier and Hudelot, C\\'eline and Colombo, Pierre},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
"""


# =============================================================================
# ColQwen3 Model Metadata
# =============================================================================

# ColQwen3-1.7B Turbo: ~1.7B params → 3.4 GB VRAM in bfloat16
slm_colqwen3_1_7b_turbo = ModelMeta(
    loader=partial(slm_colqwen3_loader),
    name="VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
    languages=SUPPORTED_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=1_700_000_000,
    memory_usage_mb=3400,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["ColPali"],
    reference="https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=SAUERKRAUTLM_CITATION + COLPALI_CITATION,
)

# ColQwen3-2B: ~2.2B params → 4.4 GB VRAM in bfloat16
slm_colqwen3_2b = ModelMeta(
    loader=partial(slm_colqwen3_loader),
    name="VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1",
    languages=SUPPORTED_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=2_200_000_000,
    memory_usage_mb=4400,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["ColPali"],
    reference="https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=SAUERKRAUTLM_CITATION + COLPALI_CITATION,
)

# ColQwen3-4B: ~4B params → 8 GB VRAM in bfloat16
slm_colqwen3_4b = ModelMeta(
    loader=partial(slm_colqwen3_loader),
    name="VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1",
    languages=SUPPORTED_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=8000,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["ColPali"],
    reference="https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=SAUERKRAUTLM_CITATION + COLPALI_CITATION,
)

# ColQwen3-8B: ~8B params → 16 GB VRAM in bfloat16
slm_colqwen3_8b = ModelMeta(
    loader=partial(slm_colqwen3_loader),
    name="VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1",
    languages=SUPPORTED_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=8_000_000_000,
    memory_usage_mb=16000,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["ColPali"],
    reference="https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=SAUERKRAUTLM_CITATION + COLPALI_CITATION,
)


# =============================================================================
# ColLFM2 Model Metadata
# =============================================================================

# ColLFM2-450M: ~450M params → 900 MB VRAM in bfloat16
slm_collfm2_450m = ModelMeta(
    loader=partial(slm_collfm2_loader),
    name="VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
    languages=SUPPORTED_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=450_000_000,
    memory_usage_mb=900,
    max_tokens=32768,
    embed_dim=128,
    license="https://huggingface.co/LiquidAI/LFM2-VL-450M/blob/main/LICENSE",  # LiquidAI LFM 1.0 License
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["ColPali"],
    reference="https://huggingface.co/VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=SAUERKRAUTLM_CITATION + COLPALI_CITATION,
)


# =============================================================================
# ColMinistral3 Model Metadata
# =============================================================================

# ColMinistral3-3B: ~3B params → 6 GB VRAM in bfloat16
slm_colministral3_3b = ModelMeta(
    loader=partial(slm_colministral3_loader),
    name="VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1",
    languages=SUPPORTED_LANGUAGES,
    revision="main",
    release_date="2025-01-01",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    memory_usage_mb=6000,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["ColPali"],
    reference="https://huggingface.co/VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=None,
    citation=SAUERKRAUTLM_CITATION + COLPALI_CITATION,
)
