import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)

from mteb import TaskMetadata
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, PromptType
from mteb.types._encoder_io import AudioInput, BatchedInput, TextInput


class SpeechT5Audio(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        self.device = device
        self.max_audio_length_s = max_audio_length_s

        self.asr_processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_asr",
            revision="53615c10408485422e09a12cda191a747f4bbe34",
        )
        self.asr_model = SpeechT5ForSpeechToText.from_pretrained(
            "microsoft/speecht5_asr",
            revision="53615c10408485422e09a12cda191a747f4bbe34",
        ).to(self.device)
        self.asr_model.eval()

        self.sampling_rate = self.asr_processor.feature_extractor.sampling_rate

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        import torchaudio

        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            batch_arrays = []
            for a in batch["audio"]:
                array = torch.tensor(a["array"], dtype=torch.float32)
                sr = a.get("sampling_rate", None)
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
                batch_arrays.append(array.numpy())

            if batch_arrays[0].ndim == 0:
                batch_arrays = [x.reshape(-1) for x in batch_arrays]
            elif batch_arrays[0].ndim > 1:
                batch_arrays = [x.reshape(x.size(0), -1) for x in batch_arrays]

            features = self.asr_processor(
                audio=batch_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=int(self.max_audio_length_s * self.sampling_rate),
                return_attention_mask=True,
            ).to(self.device)

            outputs = self.asr_model.speecht5.encoder(
                input_values=features.input_values,
                attention_mask=features.attention_mask,
            )
            last_hidden = outputs.last_hidden_state

            # Apply attention-masked pooling to exclude padding tokens
            batch_size, hidden_seq_len, hidden_size = last_hidden.shape
            device = last_hidden.device

            # Calculate proper hidden lengths based on input attention mask
            input_lengths = features.attention_mask.sum(dim=1)
            downsample_ratio = features.input_values.shape[1] / hidden_seq_len
            hidden_lengths = (input_lengths.float() / downsample_ratio).long()
            hidden_lengths = torch.clamp(hidden_lengths, min=0, max=hidden_seq_len)

            # Create attention mask for hidden states
            seq_range = torch.arange(hidden_seq_len, device=device).unsqueeze(0)
            hidden_attention_mask = (seq_range < hidden_lengths.unsqueeze(1)).to(
                last_hidden.dtype
            )

            # Apply masked mean pooling
            hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
            masked_embeddings = last_hidden * hidden_attention_mask
            valid_tokens = hidden_attention_mask.sum(dim=1)
            embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

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
        if "audio" not in inputs.dataset.features:
            raise ValueError("ASTWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


class SpeechT5Text(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.device = device
        self.tts_processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts",
            revision="30fcde30f19b87502b8435427b5f5068e401d5f6",
        )
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            revision="30fcde30f19b87502b8435427b5f5068e401d5f6",
        ).to(self.device)
        self.tts_model.eval()

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        """Get text embeddings using the text encoder."""
        all_embeddings = []

        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Processing text batches"
        ):
            texts = batch["text"]

            # Process text through tokenizer
            inputs = self.tts_processor(
                text=texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).to(self.device)

            outputs = self.tts_model.speecht5.encoder(
                input_values=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            last_hidden = outputs.last_hidden_state

            # Apply attention-masked pooling to exclude padding tokens
            attention_mask = (
                inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            )
            masked_embeddings = last_hidden * attention_mask
            valid_tokens = attention_mask.sum(dim=1)
            embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

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
        if "text" not in inputs.dataset.features:
            raise ValueError("ASTWrapper only supports audio inputs.")
        return self.get_text_embeddings(inputs, **kwargs)


class SpeechT2Multimodal(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        self.asr_encoder = SpeechT5Audio(
            model_name=model_name,
            revision=revision,
            device=device,
            max_audio_length_s=max_audio_length_s,
            **kwargs,
        )
        self.tts_encoder = SpeechT5Text(
            model_name=model_name,
            revision=revision,
            device=device,
            **kwargs,
        )

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
        text_embeddings = None
        audio_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.tts_encoder.encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )
        if "audio" in inputs.dataset.features:
            audio_embeddings = self.asr_encoder.encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )

        if text_embeddings is not None and audio_embeddings is not None:
            if len(text_embeddings) != len(audio_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + audio_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif audio_embeddings is not None:
            return audio_embeddings
        raise ValueError


# ASR model - Optimized for Speech Recognition tasks
speecht5_asr = ModelMeta(
    loader=SpeechT5Audio,
    name="microsoft/speecht5_asr",
    languages=["eng-Latn"],
    open_weights=True,
    revision="53615c10408485422e09a12cda191a747f4bbe34",
    release_date="2022-05-16",
    max_tokens=None,
    n_parameters=151_575_936,
    memory_usage_mb=578,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/speecht5_asr",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/SpeechT5",
    public_training_data="https://www.openslr.org/12",
    training_datasets=set(),  # {"LibriSpeech": ["train"]},
    modalities=["audio"],
)

# TTS model - Optimized for Text-to-Speech tasks
speecht5_tts = ModelMeta(
    loader=SpeechT5Text,
    name="microsoft/speecht5_tts",
    languages=["eng-Latn"],
    open_weights=True,
    revision="30fcde30f19b87502b8435427b5f5068e401d5f6",
    release_date="2022-05-16",
    max_tokens=None,
    n_parameters=146_335_465,
    memory_usage_mb=558,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/speecht5_tts",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/SpeechT5",
    public_training_data="https://www.openslr.org/12",
    training_datasets=set(),  # {"LibriTTS": ["train"]},
    modalities=["text"],
)

# Voice Conversion model - Optimized for Speech-to-Speech conversion tasks
speecht5_multimodal = ModelMeta(
    loader=SpeechT2Multimodal,
    name="microsoft/speecht5_multimodal",
    languages=["eng-Latn"],
    open_weights=True,
    revision="N/A",
    release_date="2022-05-16",
    max_tokens=None,
    n_parameters=297_911_401,  # Combined ASR + TTS parameters
    memory_usage_mb=1136,  # Combined memory usage
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/microsoft/speecht5_asr",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/SpeechT5",
    public_training_data="http://www.festvox.org/cmu_arctic/",
    training_datasets=set(),
    modalities=["audio", "text"],
)
