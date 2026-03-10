from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from mteb._requires_package import requires_image_dependencies, requires_package
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader
    from transformers.cache_utils import Cache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

QWEN3_VL_EMBEDDING_CITATION = """@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2601.04720},
  year={2026}
}"""

IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_LENGTH = 8192
DEFAULT_INSTRUCTION = "Represent the user's input."


def _build_qwen3_vl_for_embedding_class():
    """Lazily construct the custom Qwen3VLForEmbedding model class.

    This class mirrors the official ``Qwen3VLForEmbedding`` from the model
    repository scripts.  It wraps ``Qwen3VLModel`` (without the LM head)
    so that we can extract ``last_hidden_state`` directly, which is the
    behaviour intended by the model authors.
    """
    from dataclasses import dataclass

    from transformers.modeling_outputs import ModelOutput
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )

    @dataclass
    class Qwen3VLForEmbeddingOutput(ModelOutput):
        last_hidden_state: torch.FloatTensor | None = None
        attention_mask: torch.Tensor | None = None

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping: ClassVar[dict] = {}
        accepts_loss_kwargs = False

        def __init__(self, config: Qwen3VLConfig):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def get_video_features(
            self,
            pixel_values_videos: torch.FloatTensor,
            video_grid_thw: torch.LongTensor | None = None,
        ):
            return self.model.get_video_features(pixel_values_videos, video_grid_thw)

        def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            image_grid_thw: torch.LongTensor | None = None,
        ):
            return self.model.get_image_features(pixel_values, image_grid_thw)

        @property
        def language_model(self):
            return self.model.language_model

        @property
        def visual(self):
            return self.model.visual

        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ) -> tuple | Qwen3VLForEmbeddingOutput:
            # Setting to None enables image + text embeddings mode
            # More info: https://github.com/embeddings-benchmark/mteb/pull/4198/changes#r2899945802
            self.model.rope_deltas = None

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )
            return Qwen3VLForEmbeddingOutput(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )

    return Qwen3VLForEmbedding


class Qwen3VLEmbeddingWrapper(AbsEncoder):
    """Wrapper for Qwen3-VL-Embedding models.

    Uses the custom ``Qwen3VLForEmbedding`` model class (without the LM
    head) matching the official inference script at
    https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py
    to ensure identical results.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        visual_document_use_text: bool = False,
        **kwargs: Any,
    ):
        requires_image_dependencies()
        requires_package(
            self, "transformers", model_name, "pip install 'mteb[qwen-vl]'"
        )

        requires_package(
            self, "qwen_vl_utils", model_name, "pip install 'mteb[qwen-vl]'"
        )

        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        self.visual_document_use_text = visual_document_use_text
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        qwen3_vl_cls = _build_qwen3_vl_for_embedding_class()
        self.model = qwen3_vl_cls.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            **kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name, revision=revision, padding_side="right"
        )

    def _format_conversation(
        self,
        text: str | None = None,
        image: PILImage.Image | None = None,
        instruction: str | None = None,
    ) -> list[dict]:
        """Format a single input as a conversation for the model.

        Mirrors ``Qwen3VLEmbedder.format_model_input`` from the official
        script: system message carries the instruction, user message
        carries image (if any) followed by text (if any).
        """
        instruction = instruction or DEFAULT_INSTRUCTION
        instruction = instruction.strip()
        # Checks if the last character is not punctuation and appends "." then
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction = instruction + "."

        content: list[dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

        if not text and image is None:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        if image is not None:
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        if text:
            content.append({"type": "text", "text": text})

        return conversation

    def _preprocess_batch(
        self, conversations: list[list[dict]]
    ) -> dict[str, torch.Tensor]:
        """Preprocess a batch of conversations into model inputs.

        Mirrors ``Qwen3VLEmbedder._preprocess_inputs`` from the official
        script: apply chat template, extract vision info, then tokenize +
        pixel-process.
        """
        from qwen_vl_utils.vision_process import process_vision_info

        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception:
            logger.warning("Failed to process vision info, falling back to text-only.")
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}

        videos = None
        video_metadata = None
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs

    @staticmethod
    def _pooling_last(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract the embedding from the last non-padding token position.

        Identical to ``Qwen3VLEmbedder._pooling_last`` from the official
        script.
        """
        flipped = attention_mask.flip(dims=[1])
        last_positions = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @staticmethod
    def _prepare_images(raw_images: list) -> list[PILImage.Image]:
        """Convert batch images (tensors or PIL) to PIL Image objects."""
        import torchvision.transforms.functional as tv_functional
        from PIL import Image

        result = []
        for img in raw_images:
            if isinstance(img, Image.Image):
                result.append(img)
            else:
                result.append(tv_functional.to_pil_image(img.cpu()))
        return result

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        instruction = self.get_instruction(task_metadata, prompt_type)
        if not instruction:
            instruction = DEFAULT_INSTRUCTION

        from mteb.types import PromptType

        contains_text = "text" in inputs.dataset.features
        contains_image = "image" in inputs.dataset.features

        if (
            prompt_type == PromptType.document
            and not self.visual_document_use_text
            and contains_image
        ):
            contains_text = False

        if not contains_text and not contains_image:
            raise ValueError("No text or image features found in inputs.")

        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                inputs,
                disable=not show_progress_bar,
                desc="Encoding",
            ):
                batch_size = len(batch["text"] if contains_text else batch["image"])

                texts: list[str | None] = (
                    list(batch["text"]) if contains_text else [None] * batch_size
                )
                images: list[PILImage.Image | None] = (
                    self._prepare_images(batch["image"])
                    if contains_image
                    else [None] * batch_size
                )

                conversations = [
                    self._format_conversation(
                        text=t,
                        image=img,
                        instruction=instruction,
                    )
                    for t, img in zip(texts, images)
                ]

                processed = self._preprocess_batch(conversations)
                processed = {k: v.to(self.device) for k, v in processed.items()}

                outputs = self.model(**processed)
                embeddings = self._pooling_last(
                    outputs.last_hidden_state, processed["attention_mask"]
                )
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


qwen3_vl_embedding_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="2a50926d213628c727f38025982a76f655673f54",
    release_date="2026-01-08",
    modalities=["image", "text"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
)

qwen3_vl_embedding_8b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-8B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="418d03899aeb9d954e776dfa762006616519a914",
    release_date="2026-01-08",
    modalities=["image", "text"],
    n_parameters=8_144_793_840,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=30518,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
)
