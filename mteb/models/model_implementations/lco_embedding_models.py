import logging
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
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
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError(
                "The 'qwen_omni_utils' package is required for this model. "
                "Please install it or ensure it is in your python path."
            )

        # Pre-calculate max samples once
        max_samples = int(self.max_audio_length_seconds * self.sampling_rate)

        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            messages_batch = []
            audio_list = batch.get("audio", [])
            text_list = batch.get("text", [])
            batch_size = max(len(audio_list), len(text_list))

            for i in range(batch_size):
                content = []

                audio_row = audio_list[i] if i < len(audio_list) else None

                if audio_row is not None:
                    array = torch.tensor(audio_row["array"], dtype=torch.float32)
                    sr = audio_row.get("sampling_rate", self.sampling_rate)

                    # --- Handle empty audio if there's any ---
                    if array.numel() == 0:
                        logger.warning(
                            f"Encountered empty audio in {hf_subset}. Using 0.1s silence placeholder."
                        )
                        # Create minimal silent audio (0.1 seconds) to prevent pooling crash
                        array = torch.zeros(
                            int(self.sampling_rate * 0.1), dtype=torch.float32
                        )
                    # --------------------------------

                    if sr != self.sampling_rate:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sr, new_freq=self.sampling_rate
                        )
                        array = resampler(array)
                    if len(array) > max_samples:
                        array = array[:max_samples]
                    content.append({"type": "audio", "audio": array.numpy()})

                text_row = text_list[i] if i < len(text_list) else None
                if text_row is not None:
                    content.append({"type": "text", "text": text_row})

                # Append the training prompt
                prompt_suffix = (
                    "\nSummarize the above audio in one word:"
                    if audio_row
                    else "\nSummarize the above text in one word:"
                )
                content.append({"type": "text", "text": prompt_suffix})

                messages_batch.append([{"role": "user", "content": content}])

            text_prompts = self.processor.apply_chat_template(
                messages_batch, tokenize=False, add_generation_prompt=True
            )

            audio_inputs, _, _ = process_mm_info(
                messages_batch, use_audio_in_video=False
            )

            processor_inputs = self.processor(
                text=text_prompts,
                audio=audio_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **processor_inputs, output_hidden_states=True, return_dict=True
                )

                embeddings = outputs.hidden_states[-1][:, -1, :]

                all_embeddings.append(embeddings.cpu().to(torch.float32))

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
    memory_usage_mb=8978,
    embed_dim=2048,
    license="mit",
    reference="https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-3B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio", "text"],
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
    memory_usage_mb=17043,
    embed_dim=3584,
    license="mit",
    reference="https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio", "text"],
)
