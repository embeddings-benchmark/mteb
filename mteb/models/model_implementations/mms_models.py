from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, PromptType


class MMSWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        target_lang: str = "eng",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.target_lang = target_lang
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

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

    def _process_audio(self, audio) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(
        self, batch: Array | Iterable[tuple[Array, str]]
    ) -> list[torch.Tensor]:
        import torchaudio

        waveforms = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                waveforms.append(self._convert_audio_from_numpy(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        if isinstance(audio, np.ndarray):
                            audio = torch.from_numpy(audio).float()
                        elif isinstance(audio, list):
                            audio = torch.tensor(audio).float()
                        else:
                            audio = audio.float()
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

    def _convert_audio_from_numpy(self, audio: Array) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        import torchaudio

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
        audio,
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

                # Let feature extractor handle all padding
                batch_numpy = [
                    b.cpu().numpy() if isinstance(b, torch.Tensor) else b for b in batch
                ]

                inputs = self.feature_extractor(
                    batch_numpy,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=int(self.max_audio_length_seconds * self.sampling_rate),
                    return_attention_mask=True,
                ).to(self.device)

                outputs = self.model(
                    inputs.input_values,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                last_hidden_state = outputs.last_hidden_state

                # Apply attention-masked pooling to exclude padding tokens
                batch_size, hidden_seq_len, hidden_size = last_hidden_state.shape
                device = last_hidden_state.device

                # Calculate proper hidden lengths based on input attention mask
                input_lengths = inputs.attention_mask.sum(dim=1)
                downsample_ratio = inputs.input_values.shape[1] / hidden_seq_len
                hidden_lengths = (input_lengths.float() / downsample_ratio).long()
                hidden_lengths = torch.clamp(hidden_lengths, min=0, max=hidden_seq_len)

                # Create attention mask for hidden states
                seq_range = torch.arange(hidden_seq_len, device=device).unsqueeze(0)
                hidden_attention_mask = (seq_range < hidden_lengths.unsqueeze(1)).to(
                    last_hidden_state.dtype
                )

                # Apply masked mean pooling
                hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
                masked_embeddings = last_hidden_state * hidden_attention_mask
                valid_tokens = hidden_attention_mask.sum(dim=1)
                embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(min=1e-9)

                all_embeddings.append(embeddings.cpu())

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.zeros((0, self.model.config.hidden_size))

    def encode(
        self,
        inputs,
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
    training_datasets=set(),
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
    training_datasets=set(),
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
    training_datasets=set(),
    modalities=["audio"],
)
