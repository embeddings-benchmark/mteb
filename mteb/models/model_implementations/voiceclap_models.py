from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator
from mteb.models.model_meta import ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput


class VoiceCLAPSmallWrapper(AbsEncoder):
    """Wrapper for the custom-code VoiceCLAP-Small voice-text model.

    The model exposes dedicated ``encode_text`` and ``encode_waveform`` methods
    rather than the SentenceTransformer API, so text and audio are embedded
    separately and fused when both modalities are present.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.model = (
            AutoModel.from_pretrained(
                model_name, revision=revision, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # VoiceCLAP-Small expects 16 kHz mono waveforms.
        self.sampling_rate = 16000

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Processing text batches"
        ):
            enc = self.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                features = self.model.encode_text(
                    enc["input_ids"], enc["attention_mask"]
                )
            text_embeddings.append(features.cpu().detach().float().numpy())
        return np.vstack(text_embeddings)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)

        all_features = []
        for batch in tqdm(inputs, disable=not show_progress_bar):
            for audio in batch["audio"]:
                waveform = (
                    torch.from_numpy(np.asarray(audio["array"])).float().to(self.device)
                )
                with torch.no_grad():
                    features = self.model.encode_waveform(waveform)
                all_features.append(features.cpu().detach().float().numpy())
        return np.vstack(all_features)

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
                    "The number of texts and audios must have the same length"
                )
            return text_embeddings + audio_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif audio_embeddings is not None:
            return audio_embeddings
        raise ValueError


_VOICECLAP_CITATION = """
@misc{voicenet2026voiceclap,
      title={VoiceCLAP: Voice-Text Contrastive Embeddings},
      author={VoiceNet},
      year={2026},
      url={https://huggingface.co/VoiceNet/voiceclap-large},
}
"""

voiceclap_large = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={"trust_remote_code": True},
    name="VoiceNet/voiceclap-large",
    languages=["eng-Latn"],
    revision="361141a44121a924b4ecc3c165c65c28c5b7df26",
    release_date="2026-05-06",
    modalities=["text", "audio"],
    n_parameters=8_999_313_920,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=34330,
    max_tokens=32768,
    embed_dim=3584,
    license="cc-by-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/VoiceNet/voiceclap-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from="LCO-Embedding/LCO-Embedding-Omni-7B",
    citation=_VOICECLAP_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

voiceclap_small = ModelMeta(
    loader=VoiceCLAPSmallWrapper,
    name="VoiceNet/voiceclap-small",
    languages=["eng-Latn"],
    revision="5e016951cd69d08daf5550e408a44de57579ede9",
    release_date="2026-05-06",
    modalities=["text", "audio"],
    n_parameters=113_170_898,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=432,
    max_tokens=512,
    embed_dim=768,
    license="cc-by-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/VoiceNet/voiceclap-small",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=_VOICECLAP_CITATION,
)
