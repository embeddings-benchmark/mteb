from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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

        # Standard feature extractor used by audio models
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, revision=model_revision
        )

        # Load model
        self.model = Wav2Vec2Model.from_pretrained(
            model_name, revision=model_revision, ignore_mismatched_sizes=True
        ).to(self.device)

        # Load language adapter if available
        try:
            self.model.load_adapter(target_lang)
        except Exception:
            pass

        self.model.eval()
        self.sampling_rate = self.feature_extractor.sampling_rate

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
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size), disable=not show_progress_bar):
                batch = processed_audio[i : i + batch_size]

                batch_tensor = self._pad_audio_batch(batch)

                if batch_tensor.ndim == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                elif batch_tensor.ndim > 2:
                    batch_tensor = batch_tensor.view(batch_tensor.size(0), -1)

                inputs = self.feature_extractor(
                    batch_tensor.cpu().numpy(),
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=30 * self.sampling_rate,  # 30 seconds max
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
        raise ValueError("MMSW models only support audio encoding.")


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
    revision="3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
    release_date="2023-05-22",
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
    revision="d483345545bea550895b1aa0c6ba40236b9f1e22",
    release_date="2023-05-22",
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
    revision="1fdc004dff8c399df6b15f136abfe8e83e073d51",
    release_date="2023-05-22",
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
