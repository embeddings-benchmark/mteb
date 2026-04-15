from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator
from mteb._requires_package import requires_audio_dependencies, requires_package
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        requires_package(
            self, "qwen_omni_utils", model_name, "pip install mteb[qwen_omni_utils]"
        )
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = 10

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.processor.tokenizer.padding_side = "left"

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, **kwargs
        ).to(self.device)
        self.model.eval()

        # Audio sampling rate target
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        # Pre-calculate max samples once
        self.max_samples = int(self.max_audio_length_seconds * self.sampling_rate)

    @staticmethod
    def _prompt_suffix(batch: BatchedInput) -> str:
        """Return the prompt suffix based on the primary modality in the batch."""
        for modality in ("audio", "video", "image"):
            if batch.get(modality):
                return f"\nSummarize the above {modality} in one word:"
        return "\nSummarize the above text in one word:"

    def _build_messages(self, batch: BatchedInput) -> list[list[dict]]:
        """Build chat messages for each item in the batch."""
        audio_list = batch.get("audio", [])
        text_list = batch.get("text", [])
        video_list = batch.get("video", [])
        image_list = batch.get("image", [])
        batch_size = max(
            len(audio_list), len(text_list), len(video_list), len(image_list)
        )
        suffix = self._prompt_suffix(batch)

        messages_batch = []
        for i in range(batch_size):
            content: list[dict] = []

            # Audio
            audio_row = audio_list[i] if i < len(audio_list) else None
            if audio_row is not None:
                array = AudioCollator.resample_audio(
                    {"audio": audio_row}, self.sampling_rate, self.max_samples
                )
                content.append({"type": "audio", "audio": array})

            # Video
            video_row = video_list[i] if i < len(video_list) else None
            if video_row is not None:
                content.append({"type": "video", "video": video_row})

            # Image
            image_row = image_list[i] if i < len(image_list) else None
            if image_row is not None:
                content.append({"type": "image", "image": image_row})

            # Text
            text_row = text_list[i] if i < len(text_list) else None
            if text_row is not None:
                content.append({"type": "text", "text": text_row})

            content.append({"type": "text", "text": suffix})
            messages_batch.append([{"role": "user", "content": content}])

        return messages_batch

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
        """Encode batches of text, audio, image, or video inputs.

        Follows the batch encoding examples from
        https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B
        """
        from qwen_omni_utils import process_mm_info

        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            messages_batch = self._build_messages(batch)

            text_prompts = self.processor.apply_chat_template(
                messages_batch, tokenize=False, add_generation_prompt=True
            )

            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages_batch, use_audio_in_video=False
            )

            processor_inputs = self.processor(
                text=text_prompts,
                audio=audio_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **processor_inputs, output_hidden_states=True, return_dict=True
                )
                embeddings = outputs.hidden_states[-1][:, -1, :]
                all_embeddings.append(embeddings.cpu().to(torch.float32))

            del processor_inputs, outputs
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).numpy()


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
