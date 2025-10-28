import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MCTCTFeatureExtractor, MCTCTModel

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput

COMMON_VOICE_LANGUAGES = [
    "abk-Cyrl",  # Abkhaz
    "ara-Arab",  # Arabic
    "asm-Beng",  # Assamese
    "eus-Latn",  # Basque
    "bre-Latn",  # Breton
    "cat-Latn",  # Catalan
    "zho-Hans",  # Chinese (China)
    "zho-Hant",  # Chinese (Hong Kong) + (Taiwan)
    "chv-Cyrl",  # Chuvash
    "ces-Latn",  # Czech
    "div-Thaa",  # Dhivehi
    "nld-Latn",  # Dutch
    "eng-Latn",  # English
    "epo-Latn",  # Esperanto
    "est-Latn",  # Estonian
    "fin-Latn",  # Finnish
    "fra-Latn",  # French
    "fry-Latn",  # Frisian
    "kat-Geor",  # Georgian
    "deu-Latn",  # German
    "ell-Grek",  # Greek
    "cfm-Latn",  # Hakha Chin
    "hin-Deva",  # Hindi
    "hun-Latn",  # Hungarian
    "ind-Latn",  # Indonesian
    "ina-Latn",  # Interlingua
    "gle-Latn",  # Irish
    "ita-Latn",  # Italian
    "jpn-Jpan",  # Japanese
    "kab-Latn",  # Kabyle
    "kin-Latn",  # Kinyarwanda
    "kir-Cyrl",  # Kyrgyz
    "lav-Latn",  # Latvian
    "lit-Latn",  # Lithuanian
    "lug-Latn",  # Luganda
    "mlt-Latn",  # Maltese
    "mon-Cyrl",  # Mongolian
    "ori-Orya",  # Odia
    "fas-Arab",  # Persian
    "pol-Latn",  # Polish
    "por-Latn",  # Portuguese
    "pan-Guru",  # Punjabi
    "ron-Latn",  # Romanian
    "roh-Latn",  # Romansh Sursilvan + Romansh Vallader
    "rus-Cyrl",  # Russian
    "sah-Cyrl",  # Sakha
    "slv-Latn",  # Slovenian
    "hsb-Latn",  # Upper Sorbian
    "spa-Latn",  # Spanish
    "swe-Latn",  # Swedish
    "tam-Taml",  # Tamil
    "tat-Cyrl",  # Tatar
    "tha-Thai",  # Thai
    "tur-Latn",  # Turkish
    "ukr-Cyrl",  # Ukrainian
    "vie-Latn",  # Vietnamese
    "vot-Latn",  # Votic
    "cym-Latn",  # Welsh
]


class MCTCTWrapper(AbsEncoder):
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

        self.model = MCTCTModel.from_pretrained(model_name, revision=revision).to(
            device
        )
        self.model.eval()
        self.feature_extractor = MCTCTFeatureExtractor.from_pretrained(
            model_name, revision=revision
        )
        self.sampling_rate = self.feature_extractor.sampling_rate  # 16000 Hz

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

                # Convert to mono if needed (MCTCT expects mono audio)
                if array.dim() > 1 and array.shape[0] > 1:
                    array = torch.mean(array, dim=0, keepdim=True)

                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sampling_rate
                    )
                    array = resampler(array)

                audio_arrays.append(array.squeeze().numpy())

            feature_inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.max_audio_length_seconds * self.sampling_rate),
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_features=feature_inputs.input_features,
                    attention_mask=feature_inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                last_hidden = outputs.hidden_states[-1]

                # Apply attention-masked pooling to exclude padding tokens
                batch_size, hidden_seq_len, hidden_size = last_hidden.shape
                device = last_hidden.device

                # Calculate proper hidden lengths based on input attention mask
                input_lengths = feature_inputs.attention_mask.sum(dim=1)
                downsample_ratio = (
                    feature_inputs.attention_mask.shape[1] / hidden_seq_len
                )
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
            raise ValueError("MCTCTWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


mctct_large = ModelMeta(
    loader=MCTCTWrapper,
    name="speechbrain/m-ctc-t-large",
    languages=COMMON_VOICE_LANGUAGES,  # Supports 60 languages
    open_weights=True,
    revision="ed014c8255cea2c36f87a71cf2533b665ba00863",
    release_date="2022-01-10",
    max_tokens=None,
    n_parameters=1_058_978_691,
    memory_usage_mb=4039,
    embed_dim=1536,
    license="apache-2.0",
    reference="https://huggingface.co/speechbrain/m-ctc-t-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/speechbrain/speechbrain",
    public_training_data="https://github.com/speechbrain/speechbrain",
    training_datasets={"Common Voice", "VoxPopuli"},
    modalities=["audio"],
)
