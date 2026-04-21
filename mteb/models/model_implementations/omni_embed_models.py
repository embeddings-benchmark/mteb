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
        max_audio_length: int | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = None,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_audio_dependencies()
        requires_package(self, "peft", model_name, "pip install 'mteb[peft]'")
        from transformers import (
            AutoProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        base_model = "Qwen/Qwen2.5-Omni-7B"

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_length = max_audio_length
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        self.model = (
            Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
            .to(self.device)
            .eval()
        )

        self.processor = AutoProcessor.from_pretrained(base_model)
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    @staticmethod
    def _build_messages(
        batch: BatchedInput,
    ) -> list[list[dict[str, Any]]]:
        """Build chat messages from a batch for apply_chat_template."""
        texts = batch.get("text", [])
        images = batch.get("image", [])
        audios = batch.get("audio", [])
        videos = batch.get("video", [])
        batch_size = max(len(texts), len(images), len(audios), len(videos))

        messages = []
        for i in range(batch_size):
            content: list[dict[str, Any]] = []
            if i < len(videos) and videos[i] is not None:
                content.append({"type": "video", "video": "placeholder"})
            if i < len(audios) and audios[i] is not None:
                content.append({"type": "audio", "audio": "placeholder"})
            if i < len(images) and images[i] is not None:
                content.append({"type": "image", "image": "placeholder"})
            content.append({"type": "text", "text": texts[i] if i < len(texts) else ""})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        """Encode a single batch into normalized embeddings."""
        messages = self._build_messages(batch)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            + "<|endoftext|>"
            for msg in messages
        ]

        videos = batch.get("video")
        images = batch.get("image")
        audios = batch.get("audio")
        if audios:
            audios = [
                a["array"] if isinstance(a, dict) and "array" in a else a
                for a in audios
            ]

        model_inputs = self.processor(
            text=texts,
            audio=audios or None,
            images=images or None,
            videos=videos or None,
            return_tensors="pt",
            padding="longest",
            text_kwargs={"truncation": True, "max_length": 32768},
            videos_kwargs={
                "do_sample_frames": False,
                "use_audio_in_video": False,
            },
            audio_kwargs={"max_length": self.max_audio_length},
        ).to(self.device)

        cache_position = torch.arange(
            0, model_inputs["input_ids"].shape[1], device=self.device
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            **model_inputs, use_cache=True, cache_position=cache_position
        )
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
        has_video = "video" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features
        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                max_samples=self.max_audio_length,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_audio_length,
            )

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            all_embeddings.append(self._encode_batch(batch).cpu())

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
    n_parameters=8_931_813_888,
    n_embedding_parameters=544_997_376,
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
