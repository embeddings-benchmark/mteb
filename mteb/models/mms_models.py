from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2Model

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class MMSWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        model_revision: str = None,
        target_lang: str = "eng",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.target_lang = target_lang
        self.device = device

        # MMS models use AutoProcessor which handles language-specific processing
        self.processor = AutoProcessor.from_pretrained(
            model_name, target_lang=target_lang, revision=model_revision
        )
        
        # Load model with specified language
        self.model = Wav2Vec2Model.from_pretrained(
            model_name, revision=model_revision
        ).to(self.device)
        
        # Load language adapter if available
        try:
            self.model.load_adapter(target_lang)
        except Exception:
            pass
            
        self.model.eval()
        self.sampling_rate = 16000  # MMS models use 16kHz sampling rate

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

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                waveforms.append(self._convert_audio_from_numpy(audio))
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
                        waveforms.append(self._convert_audio_from_numpy(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio_from_numpy(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio_from_numpy(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _pad_audio_batch(self, batch):
        batch = [x.reshape(-1) if x.ndim == 0 else x for x in batch]
        max_length = max(audio.shape[0] for audio in batch)
        padded_batch = [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[0]))
            for audio in batch
        ]
        return torch.stack(padded_batch)

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size)):
                batch = processed_audio[i : i + batch_size]

                batch_tensor = self._pad_audio_batch(batch)

                if batch_tensor.ndim == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                elif batch_tensor.ndim > 2:
                    batch_tensor = batch_tensor.view(batch_tensor.size(0), -1)

                inputs = self.processor(
                    batch_tensor.cpu().numpy(),
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.model(
                    inputs.input_values,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                last_hidden_state = outputs.last_hidden_state
                embeddings = torch.mean(last_hidden_state, dim=1)
                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.model.config.hidden_size))

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_audio_embeddings(inputs, task_name=task_name, **kwargs).numpy()


# Model variants using ModelMeta
mms_300m = ModelMeta(
    loader=partial(
        MMSWrapper,
        model_name="facebook/mms-300m",
    ),
    name="facebook/mms-300m",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4ee317ce793c53dbc041fc4376c7558292dd38dc",
    release_date="2023-05-22",  # Release date of the paper
    max_tokens=None,
    n_parameters=300_000_000,
    memory_usage_mb=1210,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/mms-300m",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
    public_training_data="https://github.com/facebookresearch/fairseq/tree/main/examples/mms#data",
    training_datasets={},
    modalities=["audio"],
)

mms_1b = ModelMeta(
    loader=partial(
        MMSWrapper,
        model_name="facebook/mms-1b",
    ),
    name="facebook/mms-1b",
    languages=["eng-Latn"],
    open_weights=True,
    revision="99aa7c40c50aa514c81cfa705ae05f9a10f42fc1",
    release_date="2023-05-22",  # Release date of the paper
    max_tokens=None,
    n_parameters=1_000_000_000,
    memory_usage_mb=3683,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/mms-1b",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
    public_training_data="https://github.com/facebookresearch/fairseq/tree/main/examples/mms#data",
    training_datasets={},
    modalities=["audio"],
)

mms_1b_all = ModelMeta(
    loader=partial(
        MMSWrapper,
        model_name="facebook/mms-1b-all",
    ),
    name="facebook/mms-1b-all",
    languages=[
        "eng-Latn",
        "fra-Latn",
        "deu-Latn",
        "spa-Latn",
        "ara-Arab",
        "cmn-Hans",
        "rus-Cyrl",
    ],  # Supports 1162 languages
    open_weights=True,
    revision="b97581507fd06e35d0840faec611305a1c179f8c",
    release_date="2023-05-22",  # Release date of the paper
    max_tokens=None,
    n_parameters=1_000_000_000,
    memory_usage_mb=3680,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/mms-1b-all",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
    public_training_data="https://github.com/facebookresearch/fairseq/tree/main/examples/mms#data",
    training_datasets={},
    modalities=["audio"],
)

mms_1b_fl102 = ModelMeta(
    loader=partial(
        MMSWrapper,
        model_name="facebook/mms-1b-fl102",
    ),
    name="facebook/mms-1b-fl102",
    languages=["eng-Latn"],  # Supports 102 languages
    open_weights=True,
    revision="f65f86f27881e9fa8e382b9bb357b9b858bd20a8",
    release_date="2023-05-22",  # Release date of the paper
    max_tokens=None,
    n_parameters=1_000_000_000,
    memory_usage_mb=3680,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/mms-1b-fl102",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
    public_training_data="https://github.com/facebookresearch/fairseq/tree/main/examples/mms#data",
    training_datasets={},
    modalities=["audio"],
)

mms_1b_l1107 = ModelMeta(
    loader=partial(
        MMSWrapper,
        model_name="facebook/mms-1b-l1107",
    ),
    name="facebook/mms-1b-l1107",
    languages=["eng-Latn"],  # Supports 1107 languages
    open_weights=True,
    revision="c5d2815e9460a3acd9694ab78580d4985a00e01d",
    release_date="2023-05-22",  # Release date of the paper
    max_tokens=None,
    n_parameters=1_000_000_000,
    memory_usage_mb=3680,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/mms-1b-l1107",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
    public_training_data="https://github.com/facebookresearch/fairseq/tree/main/examples/mms#data",
    training_datasets={},
    modalities=["audio"],
)
