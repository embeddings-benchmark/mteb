from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class SeamlessM4TWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.model = SeamlessM4Tv2Model.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms = []

        if isinstance(batch, tuple):
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        audio = (
                            torch.from_numpy(audio).float()
                            if isinstance(audio, np.ndarray)
                            else audio.float()
                        )
                        if item["sampling_rate"] != self.sampling_rate:
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.sampling_rate
                            )
                            audio = resampler(audio)
                        waveforms.append(self._convert_audio(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        audio = audio.squeeze()
        
        # If audio is empty after squeeze, create a minimal valid tensor
        if audio.numel() == 0:
            # Return a very short silence instead of empty tensor
            audio = torch.zeros(160)  # 0.01 seconds at 16kHz
            
        return audio

    def _load_audio_file(self, path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                waveform = resampler(waveform)
            
            waveform = waveform.squeeze()
            
            # If loaded audio is empty, create minimal valid tensor
            if waveform.numel() == 0:
                print(f"Warning: Empty audio file {path}, using silence")
                waveform = torch.zeros(160)  # 0.01 seconds at 16kHz
                
            return waveform
            
        except Exception as e:
            print(f"Warning: Failed to load audio file {path}: {e}, using silence")
            return torch.zeros(160)  # 0.01 seconds at 16kHz

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(processed_audio), batch_size),
                disable=not show_progress_bar,
            ):
                batch = processed_audio[i : i + batch_size]

                # Filter and validate audio tensors before processing
                batch_inputs = []
                valid_indices = []
                
                for idx, audio_tensor in enumerate(batch):
                    # Check if tensor is empty or has no data
                    if audio_tensor.numel() == 0 or audio_tensor.shape[0] == 0:
                        # Skip empty tensors - don't add to batch_inputs
                        print(f"Warning: Skipping empty audio tensor at index {idx}")
                        continue
                    
                    audio_np = (
                        audio_tensor.numpy()
                        if isinstance(audio_tensor, torch.Tensor)
                        else audio_tensor
                    )
                    
                    # Double-check numpy array is not empty
                    if audio_np.size == 0:
                        print(f"Warning: Skipping empty numpy array at index {idx}")
                        continue
                        
                    batch_inputs.append(audio_np)
                    valid_indices.append(idx)

                # Skip this batch if no valid audio
                if not batch_inputs:
                    print(f"Warning: Batch {i//batch_size + 1} has no valid audio, skipping")
                    continue

                inputs = self.processor(
                    audios=batch_inputs,  # Only valid, non-empty arrays
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(self.max_audio_length_seconds * self.sampling_rate),
                ).to(self.device)

                # Get encodings through the encoder
                outputs = self.model.speech_encoder(
                    inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                # Use last hidden state for embeddings
                last_hidden_state = outputs.last_hidden_state
                embeddings = torch.mean(last_hidden_state, dim=1)
                all_embeddings.append(embeddings.cpu())

                # Clear GPU cache to prevent memory issues (like BirdCLEF OOM)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            # Return empty tensor with correct embedding dimension (like AST)
            return torch.zeros((0, self.model.config.hidden_size))

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        raise ValueError("SeamlessM4T models only support audio encoding.")


seamless_m4t_v2_large = ModelMeta(
    loader=partial(
        SeamlessM4TWrapper,
        model_name="facebook/seamless-m4t-v2-large",
        max_audio_length_seconds=30.0,  # Configurable like AST
    ),
    name="facebook/seamless-m4t-v2-large",
    languages=[
        "eng-Latn"
    ],  # multilingual: supported languages can be found in the reference
    open_weights=True,
    revision="5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
    release_date="2023-11-06",
    max_tokens=None,
    n_parameters=2_300_000_000,
    memory_usage_mb=8809,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/seamless-m4t-v2-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/seamless_communication",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
