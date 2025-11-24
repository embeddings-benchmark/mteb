import logging
import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class Qwen2AudioWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()

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
        import torchaudio

        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            audio_arrays = []
            texts = []
            audio_list = batch.get("audio", [])
            text_list = batch.get("text", [])
            batch_size = max(len(audio_list), len(text_list))

            for i in range(batch_size):
                cur_text = ""
                audio_row = audio_list[i] if i < len(audio_list) else None
                if audio_row is not None:
                    array = torch.tensor(audio_row["array"], dtype=torch.float32)
                    sr = audio_row.get("sampling_rate", None)
                    if sr is None:
                        warnings.warn(
                            f"No sampling_rate provided for an audio sample. "
                            f"Assuming {self.sampling_rate} Hz (model default)."
                        )
                        sr = self.sampling_rate

                    if sr != self.sampling_rate:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sr, new_freq=self.sampling_rate
                        )
                        array = resampler(array)
                    cur_text += "<|audio_bos|><|AUDIO|><|audio_eos|>"
                    audio_arrays.append(array.numpy())

                text_row = text_list[i] if i < len(text_list) else None
                if text_row is not None:
                    cur_text += text_row
                texts.append(cur_text)

            processor_inputs = self.processor(
                text=texts,
                audio=audio_arrays if len(audio_arrays) > 0 else None,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.max_audio_length_seconds * self.sampling_rate),
            )

            processor_inputs = {
                k: v.to(self.device) for k, v in processor_inputs.items()
            }
            with torch.no_grad():
                outputs = self.model(
                    **processor_inputs,
                    output_hidden_states=True,
                )

                hidden = outputs.hidden_states[-1]
                mask = processor_inputs["attention_mask"]

                # last non-pad index per item
                last_idx = mask.sum(dim=1) - 1
                last_idx = last_idx.clamp(min=0)

                # gather last-token embeddings
                batch_indices = torch.arange(hidden.size(0), device=self.device)
                embeddings = hidden[batch_indices, last_idx]

                all_embeddings.append(embeddings.cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()


qwen2_audio_meta = ModelMeta(
    loader=Qwen2AudioWrapper,
    name="Qwen/Qwen2-Audio-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="dd84470756e6277a71d4d7188773a43cde92696e",
    release_date="2024-08-09",
    max_tokens=131_072,
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    embed_dim=1280,
    license="mit",
    reference="https://huggingface.co/Qwen/Qwen2-Audio-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio", "text"],
)
