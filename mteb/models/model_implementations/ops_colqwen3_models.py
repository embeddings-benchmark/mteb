from typing import List, Optional, Tuple, Union

import torch
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from PIL import Image
from torch import nn
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor

from mteb._requires_package import requires_package
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .colpali_models import ColPaliEngineWrapper


class OpsColQwen3(Qwen3VLModel):
    """
    OpsColQwen3 model implementation for multi-vector document retrieval.
    """

    def __init__(self, config: Qwen3VLConfig, dims: int = 320, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.hidden_size)
        self.dims = dims
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = {
                r"^base_model\.model\.(.*)": r"\1",
                r"^model\.(.*)": r"\1",
            }

        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        attention_mask = kwargs.get("attention_mask")
        has_pixel_values = "pixel_values" in kwargs and kwargs["pixel_values"] is not None

        if has_pixel_values:
            image_grid_thw = kwargs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is passed.")

            if not torch.is_tensor(image_grid_thw):
                image_grid_thw = torch.as_tensor(image_grid_thw, device=kwargs["pixel_values"].device)

            offsets = image_grid_thw.prod(dim=1)
            unpadded = [
                pixel_sequence[: int(offset.item())] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)
            ]

            if unpadded:
                kwargs["pixel_values"] = torch.cat(unpadded, dim=0)
            else:
                kwargs["pixel_values"] = None

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )

        proj = self.custom_text_proj(last_hidden_states)
        if self.dims < self.config.text_config.hidden_size:
            proj = proj[..., : self.dims]
        proj = proj / proj.norm(dim=-1, keepdim=True)

        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)

        if has_pixel_values and self.mask_non_image_embeddings and kwargs.get("input_ids") is not None:
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask

        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size

    @property
    def temporal_patch_size(self) -> int:
        return getattr(self.visual.config, "temporal_patch_size", 1)


class OpsColQwen3Processor(BaseVisualRetrieverProcessor, Qwen3VLProcessor):
    """
    Processor for OpsColQwen3.
    """

    query_prefix: str = "Query: "
    visual_prompt_prefix: str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|im_start|>assistant\n<|endoftext|>"
    query_augmentation_token: str = "<|endoftext|>"
    image_token: str = "<|image_pad|>"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(cls, *args, device_map: Optional[str] = None, **kwargs):
        instance = super().from_pretrained(*args, device_map=device_map, **kwargs)

        if "max_num_visual_tokens" in kwargs:
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 32 * 32
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        """Process a batch of PIL images."""
        images = [image.convert("RGB") for image in images]

        batch_doc = self.__call__(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        if batch_doc["pixel_values"].numel() == 0:
            return batch_doc

        offsets = batch_doc["image_grid_thw"].prod(dim=1)
        pixel_values = list(torch.split(batch_doc["pixel_values"], offsets.tolist()))
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)

        return batch_doc

    def process_queries(
        self,
        texts: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        max_length: int = 50,
        contexts: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        if texts and queries:
            raise ValueError("Only one of 'texts' or 'queries' should be provided.")
        if queries is not None:
            texts = queries
        elif texts is None:
            raise ValueError("No texts or queries provided.")
        return self.process_texts(texts=texts)

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """Process a list of texts."""
        return self(text=texts, return_tensors="pt", padding="longest")

    def score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the MaxSim score (ColBERT-like) for query and passage embeddings."""
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Compute the number of patches (n_patches_x, n_patches_y) for an image.
        """
        patch_size = self.image_processor.patch_size
        merge_size = getattr(self.image_processor, "merge_size", 1)

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """Return a boolean tensor identifying image tokens."""
        return batch_images.input_ids == self.image_token_id


class OpsColQwen3Wrapper(ColPaliEngineWrapper):  # noqa: N801
    """Wrapper for OpsColQwen3 model."""

    def __init__(
        self,
        model_name: str = "",
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from transformers.utils.import_utils import is_flash_attn_2_available

        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else None
            )

        super().__init__(
            model_name=model_name,
            model_class=OpsColQwen3,
            processor_class=OpsColQwen3Processor,
            revision=revision,
            device=device,
            attn_implementation=attn_implementation,
            **kwargs,
        )



OPS_COLQWEN3_TRAINING_DATA = {
    "VDRMultilingualRetrieval",
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "docmatix-ir",
    "HotpotQA",
    "FEVER",
    "NQ",
    "MIRACL",
    "WebInstructSub", # MathStackExchange and ScienceStackExchange only
    "MrTyDi"
}

multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

OPS_COLQWEN3_CITATION = """
@misc{ops_colqwen3_4b,
  author       = {OpenSearch-AI},
  title        = {Ops-ColQwen3: State-of-the-Art Multimodal Embedding Model for Visual Document Retrieval},
  year         = {2026},
  url          = {https://huggingface.co/OpenSearch-AI/Ops-ColQwen3-4B},
}"""

ops_colqwen3_4b = ModelMeta(
    loader=OpsColQwen3Wrapper,
    name="OpenSearch-AI/Ops-Colqwen3-4B",
    loader_kwargs=dict(
        dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    languages=multilingual_langs,
    revision="785e185e569dc587a1eeb267a8eee14e98948bd4",
    release_date="2026-01-24",
    modalities=["image", "text"],
    n_parameters=4_800_000_000,
    memory_usage_mb=9206,
    max_tokens=32768,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=OPS_COLQWEN3_TRAINING_DATA,
    citation=OPS_COLQWEN3_CITATION,
)
