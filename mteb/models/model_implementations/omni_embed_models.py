from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator, VideoCollator
from mteb._requires_package import (
    requires_audio_dependencies,
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class OmniEmbedWrapper(AbsEncoder):
    """MTEB wrapper for Tevatron OmniEmbed.

    Built on top of Qwen2.5-Omni-7B Thinker with LoRA fine-tuning.
    Supports text, image, audio, and video modalities.
    Uses last-token pooling with left padding.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        max_audio_length: int = 2_048_000,
        num_frames: int = 16,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_audio_dependencies()
        requires_package(
            self, "qwen_omni_utils", model_name, "pip install mteb[qwen_omni_utils]"
        )
        requires_package(self, "peft", model_name, "pip install peft")
        from transformers import (
            AutoProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        BASE_MODEL = "Qwen/Qwen2.5-Omni-7B"

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_length = max_audio_length
        self.num_frames = num_frames

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            **kwargs,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    @staticmethod
    def _frames_to_pil_list(frames: torch.Tensor) -> list[Any]:
        """Convert a (num_frames, C, H, W) tensor to a list of PIL images."""
        from PIL import Image

        pil_frames = []
        for frame in frames:
            np_frame = frame.permute(1, 2, 0).cpu().numpy().astype("uint8")
            pil_frames.append(Image.fromarray(np_frame))
        return pil_frames

    @staticmethod
    def _build_messages(
        batch_texts: list[str],
        batch_images: list[Any],
        batch_audio: list[Any],
        batch_video: list[Any],
    ) -> list[list[dict[str, Any]]]:
        messages = []
        batch_size = max(
            len(batch_texts), len(batch_images), len(batch_audio), len(batch_video)
        )
        for i in range(batch_size):
            text_content = batch_texts[i] if i < len(batch_texts) else ""
            image_content = batch_images[i] if i < len(batch_images) else None
            audio_content = batch_audio[i] if i < len(batch_audio) else None
            video_content = batch_video[i] if i < len(batch_video) else None

            content: list[dict[str, Any]] = []
            if video_content is not None:
                if isinstance(video_content, torch.Tensor):
                    video_content = OmniEmbedWrapper._frames_to_pil_list(video_content)
                content.append({"type": "video", "video": video_content})
            if audio_content is not None:
                content.append({"type": "audio", "audio": audio_content})
            if image_content is not None:
                content.append({"type": "image", "image": image_content})
            content.append({"type": "text", "text": text_content})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _prepare_audio(self, raw_audio: list[Any]) -> list[Any]:
        """Resample raw audio rows to the model's expected sampling rate."""
        batch_audio = []
        for audio_row in raw_audio:
            if audio_row is None:
                batch_audio.append(None)
            elif isinstance(audio_row, dict) and "array" in audio_row:
                batch_audio.append(audio_row["array"])
            else:
                array = AudioCollator.resample_audio(
                    {"audio": audio_row},
                    self.sampling_rate,
                    self.max_audio_length,
                )
                batch_audio.append(array)
        return batch_audio

    def _encode_batch(self, batch: BatchedInput, process_mm_info: Any) -> torch.Tensor:
        """Encode a single batch into normalized embeddings."""
        messages = self._build_messages(
            batch_texts=batch.get("text", []),
            batch_images=batch.get("image", []),
            batch_audio=self._prepare_audio(batch.get("audio", [])),
            batch_video=batch.get("video", []),
        )

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            + "<|endoftext|>"
            for msg in messages
        ]

        audio_inputs, image_inputs, video_inputs = process_mm_info(
            messages, use_audio_in_video=False
        )

        model_inputs = self.processor(
            text=texts,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            text_kwargs={"truncation": True, "padding": True, "max_length": 32768},
            videos_kwargs={
                "min_pixels": 32 * 14 * 14,
                "max_pixels": 64 * 28 * 28,
                "use_audio_in_video": False,
            },
            audio_kwargs={"max_length": self.max_audio_length},
        ).to(self.device)

        outputs = self.model(
            **model_inputs, output_hidden_states=True, return_dict=True
        )
        last_hidden_states = outputs.hidden_states[-1]

        # Last-token pooling
        embeddings = last_hidden_states[:, -1]
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
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
        from qwen_omni_utils import process_mm_info

        if "video" in inputs.dataset.features or "audio" in inputs.dataset.features:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                max_frames=self.num_frames,
                max_samples=self.max_audio_length,
            )

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            all_embeddings.append(self._encode_batch(batch, process_mm_info).cpu())

        return torch.cat(all_embeddings, dim=0).float()


# --- Model Metadata ---

_OMNI_EMBED_CITATION = r"""
@article{zhuang2025tevatron,
    title={Tevatron 2.0: Unified Document Retrieval Toolkit across Scale, Language, and Modality},
    author={Zhuang, Shengyao and Ma, Xueguang and Zhan, Samantha and Zhang, Crystina},
    journal={arXiv preprint arXiv:2505.02466},
    year={2025}
}
"""

omni_embed_v01 = ModelMeta(
    loader=OmniEmbedWrapper,
    name="Tevatron/OmniEmbed-v0.1",
    revision="5cfadf87cb647be9853d9d1fd7ec568466ab5e2a",
    release_date="2025-04-12",
    languages=["eng-Latn"],
    n_parameters=8_932_000_000,
    memory_usage_mb=17_036,
    max_tokens=32768,
    embed_dim=3584,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/texttron/tevatron/tree/qwenomni",
    public_training_data="https://huggingface.co/Tevatron/OmniEmbed-v0.1#training-data",
    framework=["Tevatron", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Tevatron/OmniEmbed-v0.1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={
        "NQ",
        "MSRVTTV2T",
        "AudioCapsT2ARetrieval",
        # "Tevatron/bge-ir",  # not directly in MTEB
        # "Tevatron/pixmo-docs",  # not in MTEB
        # "Tevatron/colpali",  # not in MTEB
    },
    adapted_from="Qwen/Qwen2.5-Omni-7B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=_OMNI_EMBED_CITATION,
)
