from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .bge_models import bgem3_languages

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput

_CITATION = """@misc{multi-modal-embed-2026,
  title={multi-modal-embed: Compact Multimodal Embeddings with 2DMSE},
  author={Semantic Router Team},
  year={2026},
  url={https://huggingface.co/llm-semantic-router/multi-modal-embed-small}
}"""


class _MultiModalEmbedSmall(nn.Module):
    """Standalone tri-encoder matching the HF model-card loading recipe."""

    TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    IMAGE_ENCODER = "google/siglip-base-patch16-512"
    AUDIO_ENCODER = "openai/whisper-tiny"
    EMBED_DIM = 384

    def __init__(self) -> None:
        from transformers import (
            AutoModel,
            AutoTokenizer,
            SiglipImageProcessor,
            SiglipModel,
            WhisperFeatureExtractor,
            WhisperModel,
        )

        super().__init__()
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.TEXT_ENCODER)
        self.text_encoder = AutoModel.from_pretrained(self.TEXT_ENCODER)

        self.image_processor = SiglipImageProcessor.from_pretrained(self.IMAGE_ENCODER)
        self.image_encoder = SiglipModel.from_pretrained(
            self.IMAGE_ENCODER
        ).vision_model
        self.image_proj = nn.Linear(768, self.EMBED_DIM)

        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            self.AUDIO_ENCODER
        )
        self.audio_encoder = WhisperModel.from_pretrained(self.AUDIO_ENCODER).encoder

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_image(self, images: list[Any]) -> torch.Tensor:
        inputs = self.image_processor(images=images, return_tensors="pt")
        device = next(self.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        outputs = self.image_encoder(pixel_values=pixel_values)
        embeddings = self.image_proj(outputs.pooler_output)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_audio(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        inputs = self.audio_processor(
            waveforms, sampling_rate=16_000, return_tensors="pt"
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.audio_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=-1)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.text_encoder.load_state_dict(
            {
                k.replace("text_encoder.encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("text_encoder.encoder.")
            }
        )
        self.image_encoder.load_state_dict(
            {
                k.replace("image_encoder.vision_encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("image_encoder.vision_encoder.")
            }
        )
        image_proj = {
            k.replace("image_encoder.projection.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.projection.")
        }
        if image_proj:
            self.image_proj.load_state_dict(image_proj)
        self.audio_encoder.load_state_dict(
            {
                k.replace("audio_encoder.encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("audio_encoder.encoder.")
            }
        )


class SemanticRouterMultiModalEmbedSmallWrapper(AbsEncoder):
    """Wrapper for llm-semantic-router/multi-modal-embed-small (text/image/audio)."""

    sampling_rate = 16_000
    max_audio_seconds = 30.0

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import hf_hub_download

        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_samples = int(self.max_audio_seconds * self.sampling_rate)

        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename="model.pt",
            revision=revision,
        )
        self.model = _MultiModalEmbedSmall()
        self.model.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding text"):
            with torch.inference_mode():
                emb = self.model.encode_text(batch["text"])
            embeddings.append(emb.detach().cpu().float().numpy())
        return np.vstack(embeddings)

    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding image"):
            images = [img.convert("RGB") for img in batch["image"]]
            with torch.inference_mode():
                emb = self.model.encode_image(images)
            embeddings.append(emb.detach().cpu().float().numpy())
        return np.vstack(embeddings)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(
            target_sampling_rate=self.sampling_rate,
            max_samples=self.max_audio_samples,
        )
        embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding audio"):
            waveforms = []
            for audio in batch["audio"]:
                array = np.asarray(audio["array"], dtype=np.float32)
                if array.ndim > 1:
                    array = array.mean(axis=-1)
                waveforms.append(array)
            with torch.inference_mode():
                emb = self.model.encode_audio(waveforms)
            embeddings.append(emb.detach().cpu().float().numpy())
        return np.vstack(embeddings)

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
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features
        has_audio = "audio" in features
        if not (has_text or has_image or has_audio):
            raise ValueError(
                "multi-modal-embed-small supports text, image, and/or audio inputs"
            )

        if has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_audio_samples,
            )

        show_progress_bar = kwargs.get("show_progress_bar", True)
        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            parts: list[torch.Tensor] = []
            with torch.inference_mode():
                if has_text and batch.get("text"):
                    parts.append(self.model.encode_text(batch["text"]))
                if has_image and batch.get("image"):
                    images = [img.convert("RGB") for img in batch["image"]]
                    parts.append(self.model.encode_image(images))
                if has_audio and batch.get("audio"):
                    waveforms = []
                    for audio in batch["audio"]:
                        array = np.asarray(audio["array"], dtype=np.float32)
                        if array.ndim > 1:
                            array = array.mean(axis=-1)
                        waveforms.append(array)
                    parts.append(self.model.encode_audio(waveforms))
            if not parts:
                raise ValueError(
                    f"No supported modality found in batch: {batch.keys()}"
                )
            fused = parts[0]
            for part in parts[1:]:
                fused += part
            fused = F.normalize(fused, p=2, dim=-1)
            all_embeddings.append(fused.detach().cpu().float().numpy())
        return np.vstack(all_embeddings)


class SemanticRouterMultiModalEmbedLargeWrapper(AbsEncoder):
    """Wrapper for llm-semantic-router/multi-modal-embed-large (packaged hf_st_mm)."""

    sampling_rate = 16_000
    max_audio_seconds = 30.0

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import snapshot_download

        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_samples = int(self.max_audio_seconds * self.sampling_rate)

        local_dir = Path(snapshot_download(repo_id=model_name, revision=revision))
        src_dir = str(local_dir / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from hf_st_mm.model import MultiModalSentenceEmbedder

        with (local_dir / "config.json").open(encoding="utf-8") as handle:
            cfg = json.load(handle)

        model_cfg = cfg.get("model", cfg)
        self.model = MultiModalSentenceEmbedder(
            text_encoder_name=model_cfg["text_encoder_name"],
            image_encoder_name=model_cfg["image_encoder_name"],
            audio_encoder_name=model_cfg["audio_encoder_name"],
            embedding_dim=int(model_cfg["embedding_dim"]),
            max_text_length=int(model_cfg["max_text_length"]),
        )
        state_dict = torch.load(
            local_dir / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding text"):
            with torch.inference_mode():
                emb = self.model._encode_text(batch["text"])
            embeddings.append(emb.detach().cpu().float().numpy())
        return np.vstack(embeddings)

    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding image"):
            images = [img.convert("RGB") for img in batch["image"]]
            proc = self.model.image_processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                emb = self.model._encode_image_pixel_values(proc["pixel_values"])
            embeddings.append(emb.detach().cpu().float().numpy())
        return np.vstack(embeddings)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(
            target_sampling_rate=self.sampling_rate,
            max_samples=self.max_audio_samples,
        )
        embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding audio"):
            waveforms = []
            for audio in batch["audio"]:
                array = np.asarray(audio["array"], dtype=np.float32)
                if array.ndim > 1:
                    array = array.mean(axis=-1)
                waveforms.append(array)
            proc = self.model.audio_processor(
                waveforms, sampling_rate=self.sampling_rate, return_tensors="pt"
            )
            with torch.inference_mode():
                emb = self.model._encode_audio_features(proc["input_features"])
            embeddings.append(emb.detach().cpu().float().numpy())
        return np.vstack(embeddings)

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
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features
        has_audio = "audio" in features
        if not (has_text or has_image or has_audio):
            raise ValueError(
                "multi-modal-embed-large supports text, image, and/or audio inputs"
            )

        if has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_audio_samples,
            )

        show_progress_bar = kwargs.get("show_progress_bar", True)
        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            parts: list[torch.Tensor] = []
            with torch.inference_mode():
                if has_text and batch.get("text"):
                    parts.append(self.model._encode_text(batch["text"]))
                if has_image and batch.get("image"):
                    images = [img.convert("RGB") for img in batch["image"]]
                    proc = self.model.image_processor(
                        images=images, return_tensors="pt"
                    )
                    parts.append(
                        self.model._encode_image_pixel_values(proc["pixel_values"])
                    )
                if has_audio and batch.get("audio"):
                    waveforms = []
                    for audio in batch["audio"]:
                        array = np.asarray(audio["array"], dtype=np.float32)
                        if array.ndim > 1:
                            array = array.mean(axis=-1)
                        waveforms.append(array)
                    proc = self.model.audio_processor(
                        waveforms,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                    )
                    parts.append(
                        self.model._encode_audio_features(proc["input_features"])
                    )
            if not parts:
                raise ValueError(
                    f"No supported modality found in batch: {batch.keys()}"
                )
            fused = parts[0]
            for part in parts[1:]:
                fused += part
            fused = F.normalize(fused, p=2, dim=-1)
            all_embeddings.append(fused.detach().cpu().float().numpy())
        return np.vstack(all_embeddings)


_COMMON = dict(
    open_weights=True,
    license="apache-2.0",
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    model_type=["dense"],
    modalities=["text", "image", "audio"],
    citation=_CITATION,
)

multi_modal_embed_small = ModelMeta(
    loader=SemanticRouterMultiModalEmbedSmallWrapper,
    name="llm-semantic-router/multi-modal-embed-small",
    revision="fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4",
    release_date="2026-02-05",
    languages=["eng-Latn"],
    n_parameters=120_000_000,
    memory_usage_mb=458,
    max_tokens=256,
    embed_dim=[32, 64, 128, 256, 384],
    reference="https://huggingface.co/llm-semantic-router/multi-modal-embed-small",
    training_datasets={
        # LLaVA-CC3M, COCO Captions (not in MTEB)
        "LibriSpeech",
    },
    **_COMMON,
)

multi_modal_embed_large = ModelMeta(
    loader=SemanticRouterMultiModalEmbedLargeWrapper,
    name="llm-semantic-router/multi-modal-embed-large",
    revision="e21cde3ccc414c56f504b322662f42c603a939ee",
    release_date="2026-05-03",
    languages=bgem3_languages,
    n_parameters=1_600_000_000,
    memory_usage_mb=6100,
    max_tokens=32768,
    embed_dim=768,
    reference="https://huggingface.co/llm-semantic-router/multi-modal-embed-large",
    training_datasets=set(),
    adapted_from="llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
    framework=["PyTorch", "Sentence Transformers", "Transformers"],
    **{k: v for k, v in _COMMON.items() if k != "framework"},
)
