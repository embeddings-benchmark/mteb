import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput


class MMSWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        target_lang: str = "eng",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.model_name = model_name
        self.model_revision = revision
        self.target_lang = target_lang
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        # Standard feature extractor used by audio models
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, revision=revision
        )

        # Load model
        self.model = Wav2Vec2Model.from_pretrained(
            model_name, revision=revision, ignore_mismatched_sizes=True
        ).to(self.device)

        # Load language adapter if available
        try:
            self.model.load_adapter(target_lang)
        except Exception:
            pass

        self.model.eval()
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

                last_hidden_state = outputs.last_hidden_state

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
            raise ValueError("MMSWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


mms_1b_all = ModelMeta(
    loader=MMSWrapper,
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
    loader=MMSWrapper,
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
    loader=MMSWrapper,
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
