from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class AudioFlamingoWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: float = 30.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict | None = None,
        **kwargs: Any,
    ):
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_length_seconds = max_audio_length_seconds

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)

        # Audio Flamingo uses torch.bfloat16 commonly
        if device_map is None and self.device == "cuda":
            device_map = "auto"

        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        if device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def encode(  # noqa: PLR0914
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
        all_embeddings = []

        for batch_data in tqdm(inputs, disable=not show_progress_bar):
            audio_list = batch_data.get("audio", [])
            text_list = batch_data.get("text", [])
            batch_size = max(len(audio_list), len(text_list))

            conversations = []
            for i in range(batch_size):
                content = []

                text_row = text_list[i] if i < len(text_list) else None
                if text_row:
                    content.append({"type": "text", "text": text_row})

                audio_row = audio_list[i] if i < len(audio_list) else None
                if audio_row is not None:
                    array = AudioCollator.resample_audio(
                        {"audio": audio_row},
                        target_sampling_rate=self.sampling_rate,
                    )
                    content.append({"type": "audio", "audio": array})

                conversations.append([{"role": "user", "content": content}])

            processor_inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self.model.device)

            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device,
                    dtype=torch.bfloat16,
                ),
            ):
                outputs = self.model(
                    **processor_inputs,
                    output_hidden_states=True,
                )

                hidden = outputs.hidden_states[-1]
                mask = processor_inputs["attention_mask"]

                # last non-pad index per item (handles both left and right padding)
                reversed_mask = mask.flip(dims=[1])
                first_one_reversed = reversed_mask.argmax(dim=1)
                last_idx = mask.size(1) - 1 - first_one_reversed

                # gather last-token embeddings
                batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                embeddings = hidden[batch_indices, last_idx]

                all_embeddings.append(embeddings.float().cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()


audio_flamingo_meta = ModelMeta(
    loader=AudioFlamingoWrapper,
    name="nvidia/audio-flamingo-3-hf",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7d4bae64ee29878af6504ae6f6bb3e40492838ad",
    release_date="2025-07-10",
    max_tokens=32768,
    n_parameters=8_267_215_360,
    n_embedding_parameters=543_592_448,
    memory_usage_mb=31537,
    embed_dim=3584,
    license=None,
    reference="https://huggingface.co/nvidia/audio-flamingo-3-hf",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio", "text"],
    citation="""
@misc{audioflamingo2024,
      title={Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities},
      author={NVIDIA},
      year={2024},
      url={https://arxiv.org/abs/2507.08128},
}
""",
)
