from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator
from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput

logger = logging.getLogger(__name__)


class MSClapWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str = "microsoft/msclap-2023",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        requires_package(
            self,
            "msclap",
            "pip install 'mteb[msclap]'",
        )
        from msclap import CLAP

        self.model_name = model_name
        self.device = device
        self.sampling_rate = 48000
        self.max_audio_length_s = max_audio_length_s

        if "2022" in self.model_name:
            self.version = "2022"
            self.text_length = 100
        elif "2023" in self.model_name:
            self.version = "2023"
            self.text_length = 77
        else:
            self.version = "2023"
            self.text_length = 77

        self.use_cuda = device == "cuda"
        self.model = CLAP(version=self.version, use_cuda=self.use_cuda)
        self.model.clap = self.model.clap.to(self.device)
        self.tokenizer = self.model.tokenizer

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        import soundfile as sf

        inputs.collate_fn = AudioCollator(self.sampling_rate)

        all_embeddings = []
        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            temp_files = []
            audio_arrays = [audio["array"] for audio in batch["audio"]]

            try:
                for array in audio_arrays:
                    # Write to temp file - msclap expects file paths
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_files.append(temp_file.name)
                    sf.write(temp_file.name, array, self.sampling_rate)

                with torch.no_grad():
                    # Use the official msclap API that expects file paths
                    # https://github.com/microsoft/CLAP#api
                    audio_features = self.model.get_audio_embeddings(
                        temp_files, resample=False
                    )
                    # Normalize embeddings
                    audio_features = audio_features / audio_features.norm(
                        dim=-1, keepdim=True
                    )
                    all_embeddings.append(audio_features.cpu().detach().numpy())
            finally:
                # Clean up temp files

                for f in temp_files:
                    try:
                        Path(f).unlink()
                    except OSError:
                        pass

        return np.vstack(all_embeddings)

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        text_embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Processing text batches"
        ):
            texts = batch["text"]

            features = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.text_length,
            )
            features = {k: v.to(self.device) for k, v in features.items()}

            with torch.no_grad():
                text_features = self.model.clap.caption_encoder(features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeddings.append(text_features.cpu().detach().numpy())

        return np.vstack(text_embeddings)

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
        text_embeddings = None
        audio_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "audio" in inputs.dataset.features:
            audio_embeddings = self.get_audio_embeddings(inputs, **kwargs)

        if text_embeddings is not None and audio_embeddings is not None:
            if len(text_embeddings) != len(audio_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + audio_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif audio_embeddings is not None:
            return audio_embeddings
        raise ValueError


# Microsoft CLAP Model metadata
ms_clap_2022 = ModelMeta(
    loader=MSClapWrapper,
    name="microsoft/msclap-2022",
    languages=["eng-Latn"],
    revision="no_revision",
    release_date="2022-12-01",
    modalities=["audio", "text"],
    n_parameters=196_000_000,
    memory_usage_mb=750,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="https://github.com/microsoft/CLAP",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),
    citation="""
@inproceedings{CLAP2022,
  title={Clap learning audio concepts from natural language supervision},
  author={Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
""",
)

ms_clap_2023 = ModelMeta(
    loader=MSClapWrapper,
    name="microsoft/msclap-2023",
    languages=["eng-Latn"],
    revision="no_revision",
    release_date="2023-09-01",
    modalities=["audio", "text"],
    n_parameters=160_000_000,
    memory_usage_mb=610,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="https://github.com/microsoft/CLAP",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),
    citation="""
@misc{CLAP2023,
      title={Natural Language Supervision for General-Purpose Audio Representations},
      author={Benjamin Elizalde and Soham Deshmukh and Huaming Wang},
      year={2023},
      eprint={2309.05767},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2309.05767}
}
""",
)
