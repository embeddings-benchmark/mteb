import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Model

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput

# ISO 639-3 codes for languages supported by wav2vec2 models
WAV2VEC2_LANGUAGES = [
    "afr-Latn",
    "sqi-Latn",
    "amh-Latn",
    "ara-Latn",
    "hye-Latn",
    "asm-Latn",
    "aze-Latn",
    "eus-Latn",
    "bel-Latn",
    "ben-Beng",
    "bos-Latn",
    "bre-Latn",
    "bul-Latn",
    "mya-Latn",
    "cat-Latn",
    "khm-Latn",
    "zho-Latn",
    "hrv-Latn",
    "ces-Latn",
    "dan-Latn",
    "nld-Latn",
    "eng-Latn",
    "epo-Latn",
    "est-Latn",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "kat-Latn",
    "deu-Latn",
    "ell-Latn",
    "guj-Latn",
    "hau-Latn",
    "heb-Latn",
    "hin-Deva",
    "hun-Latn",
    "isl-Latn",
    "ind-Latn",
    "gle-Latn",
    "ita-Latn",
    "jpn-Latn",
    "jav-Latn",
    "kan-Latn",
    "kaz-Latn",
    "kir-Latn",
    "abk-Cyrl",
    "bak-Cyrl",
    "ceb-Latn",
    "chv-Cyrl",
    "div-Thaa",
    "fao-Latn",
    "grn-Latn",
    "hat-Latn",
    "haw-Latn",
    "ina-Latn",
    "kin-Latn",
]


class Wav2Vec2AudioWrapper(AbsEncoder):
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

        # Try to load base model first, fallback to CTC if needed
        try:
            self.model = Wav2Vec2Model.from_pretrained(
                model_name, revision=revision
            ).to(self.device)
            self.is_ctc_model = False
        except Exception:
            # Fallback to CTC model for models that don't have base versions
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name, revision=revision
            ).to(self.device)
            self.is_ctc_model = True

        self.model.eval()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate

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
            audio_arrays = []
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
                audio_arrays.append(array.numpy())

            feature_inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=int(self.max_audio_length_seconds * self.sampling_rate),
                return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    feature_inputs.input_values,
                    attention_mask=feature_inputs.attention_mask,
                    output_hidden_states=True,
                )

                last_hidden_state = (
                    outputs.hidden_states[-1]
                    if self.is_ctc_model
                    else outputs.last_hidden_state
                )

                # last_hidden_state: [B, hidden_seq_len, hidden_size]
                batch_size, hidden_seq_len, hidden_size = last_hidden_state.shape
                device = last_hidden_state.device

                # inputs.attention_mask is per-sample mask over input_values
                input_lengths = (
                    feature_inputs.attention_mask.sum(dim=1).cpu().numpy().astype(int)
                )  # shape (B,)

                hidden_lengths = [
                    self.model._get_feat_extract_output_lengths(l)
                    for l in input_lengths
                ]
                hidden_lengths = torch.tensor(
                    hidden_lengths, device=device, dtype=torch.long
                )  # (B,)

                # clamp to be safe
                hidden_lengths = torch.clamp(hidden_lengths, min=0, max=hidden_seq_len)

                # build mask vectorized: seq_range shape [1, hidden_seq_len]
                seq_range = torch.arange(hidden_seq_len, device=device).unsqueeze(
                    0
                )  # [1, hidden_seq_len]
                hidden_attention_mask = (seq_range < hidden_lengths.unsqueeze(1)).to(
                    last_hidden_state.dtype
                )  # [B, hidden_seq_len]

                # apply masked mean pooling
                hidden_attention_mask = hidden_attention_mask.unsqueeze(
                    -1
                )  # [B, hidden_seq_len, 1]
                masked_embeddings = (
                    last_hidden_state * hidden_attention_mask
                )  # [B, hidden_seq_len, hidden_size]
                valid_tokens = hidden_attention_mask.sum(dim=1)  # [B, 1]

                # Safer division to avoid NaN
                sum_embeddings = masked_embeddings.sum(dim=1)  # [B, hidden_size]

                # Check for NaN in the sum and replace with zeros
                nan_mask = torch.isnan(sum_embeddings).any(
                    dim=1, keepdim=True
                )  # [B, 1]
                sum_embeddings = torch.where(
                    nan_mask.expand_as(sum_embeddings),
                    torch.zeros_like(sum_embeddings),
                    sum_embeddings,
                )

                valid_tokens_safe = torch.where(
                    valid_tokens > 0, valid_tokens, torch.ones_like(valid_tokens)
                )
                embeddings = sum_embeddings / valid_tokens_safe  # [B, hidden_size]

                # Zero out embeddings where we had no valid tokens
                zero_mask = (valid_tokens <= 0).expand_as(embeddings)
                embeddings = torch.where(
                    zero_mask, torch.zeros_like(embeddings), embeddings
                )

                # Final NaN check and replacement
                embeddings = torch.where(
                    torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings
                )

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
            raise ValueError("Wav2Vec2Wrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


wav2vec2_xlsr_300m = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-xls-r-300m",
    languages=WAV2VEC2_LANGUAGES,
    revision="1a640f32ac3e39899438a2931f9924c02f080a54",
    release_date="2021-10-13",
    modalities=["audio"],
    n_parameters=300_000_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/pytorch/fairseq",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-300m",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),
)

wav2vec2_xlsr_300m_phoneme = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="vitouphy/wav2vec2-xls-r-300m-phoneme",
    languages=["eng-Latn"],
    revision="bf9913bf096d133cf4eca64ed75981ebf0545c9d",
    release_date="2022-05-19",
    modalities=["audio"],
    n_parameters=300_000_000,
    memory_usage_mb=1200,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-phoneme",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)
wav2vec2_xlsr_1b = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-xls-r-1b",
    languages=WAV2VEC2_LANGUAGES,
    revision="35eaea9a0ed0f97592277d79208e40ab8917d1e3",
    release_date="2024-09-10",
    modalities=["audio"],
    n_parameters=1_000_000_000,
    memory_usage_mb=4500,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/pytorch/fairseq",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-1b",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)

wav2vec2_xlsr_2b = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-xls-r-2b",
    languages=WAV2VEC2_LANGUAGES,
    revision="3b6d89d0fabead7da552eaaa07549c7c9c36d303",
    release_date="2024-09-10",
    modalities=["audio"],
    n_parameters=2_000_000_000,
    memory_usage_mb=9000,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/pytorch/fairseq",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-2b",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)

wav2vec2_xlsr_2b_translation = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-xls-r-2b-21-to-en",
    languages=WAV2VEC2_LANGUAGES,
    revision="70239d15f5b39ecbc936a5e214bf401b7f17e210",
    release_date="2024-09-10",
    modalities=["audio"],
    n_parameters=2_000_000_000,
    memory_usage_mb=9200,
    max_tokens=float("inf"),
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/wav2vec2-xls-r-2b-21-to-en",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)


wav2vec2_base = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-base",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=362,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_base_960h = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-base-960h",
    languages=["eng-Latn"],
    open_weights=True,
    revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=360,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-base-960h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_large = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-large",
    languages=["eng-Latn"],
    open_weights=True,
    revision="312b2410566b698c7a649068d413b2067848bd75",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1_209,
    embed_dim=1_024,
    license="apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_large_xlsr_53 = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-large-xlsr-53",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c3f9d884181a224a6ac87bf8885c84d1cff3384f",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1_209,
    embed_dim=1_024,
    license="apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-large-xlsr-53",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_lv_60_espeak_cv_ft = ModelMeta(
    loader=Wav2Vec2AudioWrapper,
    name="facebook/wav2vec2-lv-60-espeak-cv-ft",
    languages=["eng-Latn"],
    open_weights=True,
    revision="ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1_209,
    embed_dim=1_024,
    license="apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
