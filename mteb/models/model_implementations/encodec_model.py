import logging
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

logger = logging.getLogger(__name__)


class EncodecWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        from transformers import AutoProcessor, EncodecModel

        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.model = EncodecModel.from_pretrained(model_name, revision=revision).to(
            device
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.sampling_rate  # 24000 Hz typically

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
            for idx, a in enumerate(batch["audio"]):
                array = torch.tensor(a["array"], dtype=torch.float32)
                sr = a.get("sampling_rate", None)

                if sr is None:
                    warnings.warn(
                        f"No sampling_rate provided for an audio sample. "
                        f"Assuming {self.sampling_rate} Hz (model default)."
                    )
                    sr = self.sampling_rate

                # Convert to mono if needed
                if array.dim() > 1 and array.shape[0] > 1:
                    array = torch.mean(array, dim=0, keepdim=True)

                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sampling_rate
                    )
                    array = resampler(array)

                array = array.squeeze()
                audio_arrays.append(array.numpy())

            with torch.no_grad():
                # Process audio through EnCodec's processor
                max_samples = int(self.max_audio_length_seconds * self.sampling_rate)

                feature_inputs = self.processor(
                    raw_audio=audio_arrays,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_samples,
                ).to(self.device)

                # Get the latent representations directly from the encoder
                latent = self.model.encoder(feature_inputs.input_values)

                # Apply mean pooling over the time dimension to get fixed-size embeddings
                embeddings = torch.mean(latent, dim=2)  # Average over time dimension

                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

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
            raise ValueError("EncodecWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


encodec_24khz = ModelMeta(
    loader=EncodecWrapper,
    name="facebook/encodec_24khz",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1dbe2ae3f1de713481a3b3e7c47f357092ee040",
    release_date="2022-10-25",
    max_tokens=None,
    n_parameters=23_273_218,
    memory_usage_mb=88,
    embed_dim=128,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/facebook/encodec_24khz",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/encodec",
    public_training_data=None,
    training_datasets=None,  # ["AudioSet", "VCTK", "DNS-Challenge"],
    modalities=["audio"],
)
