import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import WhisperModel, WhisperProcessor

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput


class WhisperAudioWrapper(AbsEncoder):
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

        self.model = WhisperModel.from_pretrained(model_name, revision=revision).to(
            device
        )
        self.model.eval()
        self.processor = WhisperProcessor.from_pretrained(model_name, revision=revision)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        hidden_layer: float = 1.0,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        import torchaudio

        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            audio_arrays = []
            for a in batch["audio"]:
                array = torch.tensor(a["array"], dtype=torch.float32)
                sr = (
                    a.get("sampling_rate")
                    if isinstance(a, dict)
                    else a["sampling_rate"]
                )
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
                audio_arrays.append(array.numpy())

            feature_inputs = self.processor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=int(self.max_audio_length_seconds * self.sampling_rate),
                return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.encoder(
                    feature_inputs.input_features,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states
                no_hidden_states = len(hidden_states)

                layer_idx = int(hidden_layer * no_hidden_states)
                layer_idx = min(max(layer_idx, 1), no_hidden_states) - 1

                selected_hidden = hidden_states[layer_idx]

                # Apply attention-masked pooling to exclude padding tokens
                if (
                    hasattr(feature_inputs, "attention_mask")
                    and feature_inputs.attention_mask is not None
                ):
                    batch_size, hidden_seq_len, hidden_size = selected_hidden.shape
                    device = selected_hidden.device

                    # For Whisper, the attention mask should match the sequence length
                    # If it doesn't, we need to calculate proper lengths
                    if feature_inputs.attention_mask.shape[1] != hidden_seq_len:
                        # Calculate downsample ratio and proper hidden lengths
                        input_lengths = feature_inputs.attention_mask.sum(dim=1)
                        downsample_ratio = (
                            feature_inputs.attention_mask.shape[1] / hidden_seq_len
                        )
                        hidden_lengths = (
                            input_lengths.float() / downsample_ratio
                        ).long()
                        hidden_lengths = torch.clamp(
                            hidden_lengths, min=0, max=hidden_seq_len
                        )

                        # Create attention mask for hidden states
                        seq_range = torch.arange(
                            hidden_seq_len, device=device
                        ).unsqueeze(0)
                        hidden_attention_mask = (
                            seq_range < hidden_lengths.unsqueeze(1)
                        ).to(selected_hidden.dtype)
                    else:
                        # Use the attention mask directly if dimensions match
                        hidden_attention_mask = feature_inputs.attention_mask.to(
                            selected_hidden.dtype
                        )

                    # Apply masked mean pooling
                    hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
                    masked_embeddings = selected_hidden * hidden_attention_mask
                    valid_tokens = hidden_attention_mask.sum(dim=1)
                    embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(
                        min=1e-9
                    )
                else:
                    # Fallback to simple mean pooling if no attention mask
                    embeddings = torch.mean(selected_hidden, dim=1)

                all_embeddings.append(embeddings.cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()

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
            raise ValueError("WhisperAudioWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


# Model Metas for Different Whisper Models
whisper_langs = [
    "eng-Latn",
    "zho-Hans",
    "deu-Latn",
    "spa-Latn",
    "rus-Cyrl",
    "kor-Hang",
    "fra-Latn",
    "jpn-Jpan",
    "por-Latn",
    "tur-Latn",
    "pol-Latn",
    "cat-Latn",
    "nld-Latn",
    "ara-Arab",
    "swe-Latn",
    "ita-Latn",
    "ind-Latn",
    "hin-Deva",
    "fin-Latn",
    "vie-Latn",
    "heb-Hebr",
    "ukr-Cyrl",
    "ell-Grek",
    "msa-Latn",
    "ces-Latn",
    "ron-Latn",
    "dan-Latn",
    "hun-Latn",
    "tam-Taml",
    "nob-Latn",
    "tha-Thai",
    "urd-Arab",
    "hrv-Latn",
    "bul-Cyrl",
    "lit-Latn",
    "lat-Latn",
    "mri-Latn",
    "mal-Mlym",
    "cym-Latn",
    "slk-Latn",
    "tel-Telu",
    "fas-Arab",
    "lav-Latn",
    "ben-Beng",
    "srp-Cyrl",
    "aze-Latn",
    "slv-Latn",
    "kan-Knda",
    "est-Latn",
    "mkd-Cyrl",
    "bre-Latn",
    "eus-Latn",
    "isl-Latn",
    "hye-Armn",
    "nep-Deva",
    "mon-Cyrl",
    "bos-Latn",
    "kaz-Cyrl",
    "sqi-Latn",
    "swa-Latn",
    "glg-Latn",
    "mar-Deva",
    "pan-Guru",
    "sin-Sinh",
    "khm-Khmr",
    "sna-Latn",
    "yor-Latn",
    "som-Latn",
    "afr-Latn",
    "oci-Latn",
    "kat-Geor",
    "bel-Cyrl",
    "tgk-Cyrl",
    "snd-Arab",
    "guj-Gujr",
    "amh-Ethi",
    "yid-Hebr",
    "lao-Laoo",
    "uzb-Latn",
    "fao-Latn",
    "hat-Latn",
    "pus-Arab",
    "tuk-Latn",
    "nno-Latn",
    "mlt-Latn",
    "san-Deva",
    "ltz-Latn",
    "mya-Mymr",
    "bod-Tibt",
    "tgl-Latn",
    "mlg-Latn",
    "asm-Beng",
    "tat-Cyrl",
    "haw-Latn",
    "lin-Latn",
    "hau-Latn",
    "bak-Cyrl",
    "jav-Latn",
    "sun-Latn",
]


whisper_tiny = ModelMeta(
    loader=WhisperAudioWrapper,
    name="openai/whisper-tiny",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=39_000_000,
    memory_usage_mb=144,
    embed_dim=512,
    license="mit",
    reference="https://huggingface.co/openai/whisper-tiny",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_base = ModelMeta(
    loader=WhisperAudioWrapper,
    name="openai/whisper-base",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=74_000_000,
    memory_usage_mb=277,
    embed_dim=512,
    license="mit",
    reference="https://huggingface.co/openai/whisper-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_small = ModelMeta(
    loader=WhisperAudioWrapper,
    name="openai/whisper-small",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=244_000_000,
    memory_usage_mb=922,
    embed_dim=768,
    license="mit",
    reference="https://huggingface.co/openai/whisper-small",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_medium = ModelMeta(
    loader=WhisperAudioWrapper,
    name="openai/whisper-medium",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=769_000_000,
    memory_usage_mb=2914,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/openai/whisper-medium",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

whisper_large_v3 = ModelMeta(
    loader=WhisperAudioWrapper,
    name="openai/whisper-large-v3",
    languages=whisper_langs,
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=1_550_000_000,
    memory_usage_mb=5887,
    embed_dim=1280,
    license="mit",
    reference="https://huggingface.co/openai/whisper-large-v3",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
