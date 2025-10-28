import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies, requires_package
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput


class CNN14Wrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        self.model_name = model_name
        self.device = device
        self.max_audio_length_s = max_audio_length_s

        requires_package(
            self,
            "speechbrain",
            "speechbrain/cnn14-esc50",
            "pip install 'mteb[speechbrain]'",
        )

        from speechbrain.inference.classifiers import AudioClassifier

        # Load the SpeechBrain model
        self.model = AudioClassifier.from_hparams(
            source=model_name,
            savedir="pretrained_models/cnn14-esc50",
            run_opts={"device": device},
        )

        # SpeechBrain uses a 16kHz sampling rate for audio
        self.sampling_rate = 16000

    def _pad_audio_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(w.shape[0] for w in batch)
        padded = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in batch]
        return torch.stack(padded)

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
            audio_tensors = []
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

                array = array.squeeze()

                # Apply audio truncation (configurable limit)
                max_length = int(self.max_audio_length_s * self.sampling_rate)
                if array.shape[-1] > max_length:
                    array = array[..., :max_length]

                audio_tensors.append(array)

            with torch.no_grad():
                # Convert batch to tensors and move to device
                batch_tensor = self._pad_audio_batch(audio_tensors).to(self.device)

                feats = self.model.mods.compute_features(batch_tensor)
                b, f, t = feats.shape
                if f < 64 or t < 80:
                    # zero-pad in the frequency or time dimension until it's at least [64, 80]
                    pad_freq = max(0, 64 - f)
                    pad_time = max(0, 80 - t)
                    feats = torch.nn.functional.pad(feats, (0, pad_time, 0, pad_freq))
                embeddings = self.model.mods.embedding_model(feats)
                # Apply mean pooling over time dimension if needed
                if embeddings.dim() > 2:
                    embeddings = torch.mean(embeddings, dim=1)

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
            raise ValueError("ASTWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


cnn14_esc50 = ModelMeta(
    loader=CNN14Wrapper,
    name="speechbrain/cnn14-esc50",
    languages=["eng-Latn"],
    open_weights=True,
    revision="422a112e9a22a5fac0d37571aacaee5caf154395",
    release_date="2022-11-26",
    max_tokens=None,
    n_parameters=80_753_615,
    memory_usage_mb=308,
    embed_dim=2048,
    license="apache-2.0",
    reference="https://huggingface.co/speechbrain/cnn14-esc50",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/speechbrain/speechbrain",
    public_training_data=None,
    training_datasets=None,  # ["ESC-50", "VGGSound"],
    modalities=["audio"],
)
