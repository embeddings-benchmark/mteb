from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator, VideoCollator
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class LCOEmbedding(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = None,
        num_frames: int | None = None,
        max_audio_length: int | None = None,
        **kwargs: Any,
    ):
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.max_audio_length = max_audio_length

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.processor.tokenizer.padding_side = "left"

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, **kwargs
        ).to(self.device)
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    @staticmethod
    def _prompt_suffix(batch: BatchedInput) -> str:
        """Return the prompt suffix based on the primary modality in the batch."""
        for modality in ("audio", "video", "image"):
            if batch.get(modality):
                return f"\nSummarize the above {modality} in one word:"
        return "\nSummarize the above text in one word:"

    @staticmethod
    def _build_messages(batch: BatchedInput, suffix: str) -> list[list[dict[str, Any]]]:
        """Build chat messages for each item in the batch."""
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
            if i < len(texts) and texts[i] is not None:
                content.append({"type": "text", "text": texts[i]})
            content.append({"type": "text", "text": suffix})
            messages.append([{"role": "user", "content": content}])
        return messages

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
            suffix = self._prompt_suffix(batch)
            messages = self._build_messages(batch, suffix)

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
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
                padding=True,
                return_tensors="pt",
                videos_kwargs={
                    "do_sample_frames": False,
                    "use_audio_in_video": False,
                },
                audio_kwargs={"max_length": self.max_audio_length},
            ).to(self.device)

            outputs = self.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )
            embeddings = outputs.hidden_states[-1][:, -1, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).float()


lco_3b = ModelMeta(
    loader=LCOEmbedding,
    name="LCO-Embedding/LCO-Embedding-Omni-3B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="eea763cfaf673e955ae86c64968896a3fea70189",
    release_date="2025-10-23",
    max_tokens=32768,
    n_parameters=4_703_464_448,
    n_embedding_parameters=311164928,
    memory_usage_mb=8978,
    embed_dim=2048,
    license="mit",
    reference="https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-3B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # SeaDoc (not in MTEB)
    ),
    modalities=["audio", "image", "text", "video"],
    citation="""
@misc{xiao2025scalinglanguagecentricomnimodalrepresentation,
  title={Scaling Language-Centric Omnimodal Representation Learning},
  author={Chenghao Xiao and Hou Pong Chan and Hao Zhang and Weiwen Xu and Mahani Aljunied and Yu Rong},
  year={2025},
  eprint={2510.11693},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.11693},
}""",
)

lco_7b = ModelMeta(
    loader=LCOEmbedding,
    name="LCO-Embedding/LCO-Embedding-Omni-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="3d38f58aae1253a4443b1270b0767f1e533936cf",
    release_date="2025-10-15",
    max_tokens=32768,
    n_parameters=8_931_813_888,
    n_embedding_parameters=544997376,
    memory_usage_mb=17043,
    embed_dim=3584,
    license="mit",
    reference="https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # SeaDoc (not in MTEB)
    ),
    modalities=["audio", "image", "text", "video"],
    citation="""
@misc{xiao2025scalinglanguagecentricomnimodalrepresentation,
  title={Scaling Language-Centric Omnimodal Representation Learning},
  author={Chenghao Xiao and Hou Pong Chan and Hao Zhang and Weiwen Xu and Mahani Aljunied and Yu Rong},
  year={2025},
  eprint={2510.11693},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.11693},
}""",
)
