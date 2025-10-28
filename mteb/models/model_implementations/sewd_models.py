import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SEWDForCTC, Wav2Vec2FeatureExtractor

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput


class SewDWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        # SewD uses the same feature extractor as Wav2Vec2
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = SEWDForCTC.from_pretrained(model_name).to(self.device)
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
                sr = a.get("sampling_rate") if isinstance(a, dict) else a["sampling_rate"]
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
                # SewD model outputs
                outputs = self.model(
                    feature_inputs.input_values,
                    attention_mask=feature_inputs.attention_mask,
                    output_hidden_states=True,
                )

                # Get embeddings from last hidden state
                last_hidden_state = outputs.hidden_states[-1]

                # Apply attention-masked pooling to exclude padding tokens
                batch_size, hidden_seq_len, hidden_size = last_hidden_state.shape
                device = last_hidden_state.device

                # Calculate proper hidden lengths based on input attention mask
                input_lengths = feature_inputs.attention_mask.sum(dim=1)
                downsample_ratio = feature_inputs.input_values.shape[1] / hidden_seq_len
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
            raise ValueError("SewDWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


sewd_base = ModelMeta(
    loader=SewDWrapper,
    name="asapp/sew-d-base-plus-400k-ft-ls100h",
    languages=["eng-Latn"],
    open_weights=True,
    revision="dba3bb02fda4248b6e082697eee756de8fe8aa8a",
    release_date="2021-09-14",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=675,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/asapp/sew-d-base-plus-400k-ft-ls100h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"LibriSpeech"},
    modalities=["audio"],
)

sewd_tiny = ModelMeta(
    loader=SewDWrapper,
    name="asapp/sew-d-tiny-100k-ft-ls100h",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1966cdcfbd2123ee90b003c2aa6ec6fe204cc4d8",
    release_date="2021-09-14",
    max_tokens=float("inf"),
    n_parameters=19_700_000,
    memory_usage_mb=92,
    embed_dim=256,
    license="apache-2.0",
    reference="https://huggingface.co/asapp/sew-d-tiny-100k-ft-ls100h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"LibriSpeech"},
    modalities=["audio"],
)

sewd_mid = ModelMeta(
    loader=SewDWrapper,
    name="asapp/sew-d-mid-400k-ft-ls100h",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b2ff9fdb3bddc81657cf5f16bc0c510be0a39b3e",
    release_date="2021-09-14",
    max_tokens=float("inf"),
    n_parameters=139_000_000,
    memory_usage_mb=530,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/asapp/sew-d-mid-400k-ft-ls100h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"LibriSpeech"},
    modalities=["audio"],
)
