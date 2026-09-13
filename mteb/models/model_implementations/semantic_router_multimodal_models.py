from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .bge_models import bgem3_languages

if TYPE_CHECKING:
    import numpy as np
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


def _remap_text_encoder_state_dict(
    state_dict: dict[str, Any], model: nn.Module
) -> dict[str, Any]:
    """Map ST Transformer keys between ``.model.`` and ``.auto_model.``."""
    model_keys = model.state_dict().keys()
    wants_auto = any(k.startswith("text_model.0.auto_model.") for k in model_keys)
    wants_model = any(k.startswith("text_model.0.model.") for k in model_keys)
    has_auto = any(k.startswith("text_model.0.auto_model.") for k in state_dict)
    has_model = any(k.startswith("text_model.0.model.") for k in state_dict)

    if wants_auto and has_model and not has_auto:
        old, new = "text_model.0.model.", "text_model.0.auto_model."
    elif wants_model and has_auto and not has_model:
        old, new = "text_model.0.auto_model.", "text_model.0.model."
    else:
        return state_dict

    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith(old):
            remapped[new + key[len(old) :]] = value
        else:
            remapped[key] = value
    return remapped


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
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.TEXT_ENCODER, revision="8b3219a92973c328a8e22fadcfa821b5dc75636a"
        )
        self.text_encoder = AutoModel.from_pretrained(
            self.TEXT_ENCODER, revision="8b3219a92973c328a8e22fadcfa821b5dc75636a"
        )

        self.image_processor = SiglipImageProcessor.from_pretrained(
            self.IMAGE_ENCODER, revision="753a949581523b60257d93e18391e8c27f72eb22"
        )
        self.image_encoder = SiglipModel.from_pretrained(
            self.IMAGE_ENCODER, revision="753a949581523b60257d93e18391e8c27f72eb22"
        ).vision_model
        self.image_proj = nn.Linear(768, self.EMBED_DIM)

        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            self.AUDIO_ENCODER, revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
        )
        self.audio_encoder = WhisperModel.from_pretrained(
            self.AUDIO_ENCODER, revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
        ).encoder

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


class _MultiModalEmbedLarge(nn.Module):
    """Standalone large tri-encoder (mmBERT + SigLIP2 + Whisper-medium)."""

    def __init__(
        self,
        text_encoder_name: str,
        image_encoder_name: str,
        audio_encoder_name: str,
        embedding_dim: int,
        max_text_length: int,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from transformers import (
            AutoModel,
            AutoProcessor,
            WhisperFeatureExtractor,
            WhisperModel,
        )

        super().__init__()
        self.text_model = SentenceTransformer(
            text_encoder_name,
            revision="c544097d7603b10c546560e8be5d7fe0965a7909",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        self.text_model.max_seq_length = max_text_length

        self.image_model = AutoModel.from_pretrained(
            image_encoder_name,
            revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
            trust_remote_code=True,
        )
        self.image_processor = AutoProcessor.from_pretrained(
            image_encoder_name,
            revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
            trust_remote_code=True,
        )

        whisper = WhisperModel.from_pretrained(
            audio_encoder_name,
            revision="abdf7c39ab9d0397620ccaea8974cc764cd0953e",
        )
        self.audio_model = whisper.encoder
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            audio_encoder_name,
            revision="abdf7c39ab9d0397620ccaea8974cc764cd0953e",
        )

        text_dim = (
            self.text_model.get_embedding_dimension()
            if hasattr(self.text_model, "get_embedding_dimension")
            else self.text_model.get_sentence_embedding_dimension()
        )
        image_dim = self._get_vision_dim(self.image_model)
        audio_dim = whisper.config.d_model

        self.text_proj = (
            nn.Linear(text_dim, embedding_dim)
            if text_dim != embedding_dim
            else nn.Identity()
        )
        self.image_proj = (
            nn.Linear(image_dim, embedding_dim)
            if image_dim != embedding_dim
            else nn.Identity()
        )
        self.audio_proj = (
            nn.Linear(audio_dim, embedding_dim)
            if audio_dim != embedding_dim
            else nn.Identity()
        )

    @staticmethod
    def _get_vision_dim(model: nn.Module) -> int:
        if hasattr(model, "vision_model") and hasattr(model.config, "vision_config"):
            return int(model.config.vision_config.hidden_size)
        if hasattr(model.config, "hidden_size"):
            return int(model.config.hidden_size)
        raise ValueError("Could not infer image hidden size")

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        if hasattr(self.text_model, "preprocess"):
            features = self.text_model.preprocess(texts)
        else:
            features = self.text_model.tokenize(texts)
        features = {
            k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()
        }
        out = self.text_model(features)
        return F.normalize(self.text_proj(out["sentence_embedding"]), p=2, dim=-1)

    def encode_image(self, images: list[Any]) -> torch.Tensor:
        proc = self.image_processor(images=images, return_tensors="pt")
        device = next(self.parameters()).device
        pixel_values = proc["pixel_values"].to(device)
        if hasattr(self.image_model, "vision_model"):
            out = self.image_model.vision_model(
                pixel_values=pixel_values, output_hidden_states=False
            )
        else:
            out = self.image_model(
                pixel_values=pixel_values, output_hidden_states=False
            )
        hidden = out.last_hidden_state
        pooled = (
            hidden[:, 1:].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
        )
        return F.normalize(self.image_proj(pooled), p=2, dim=-1)

    def encode_audio(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        proc = self.audio_processor(
            waveforms, sampling_rate=16_000, return_tensors="pt"
        )
        device = next(self.parameters()).device
        input_features = proc["input_features"].to(
            device=device, dtype=self.audio_model.conv1.weight.dtype
        )
        out = self.audio_model(
            input_features=input_features, output_hidden_states=False
        )
        pooled = out.last_hidden_state.mean(dim=1)
        return F.normalize(self.audio_proj(pooled), p=2, dim=-1)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = _remap_text_encoder_state_dict(state_dict, self)
        self.load_state_dict(state_dict)


class SemanticRouterMultiModalEmbedWrapper(AbsEncoder):
    """Shared encode paths for multi-modal-embed small/large backends."""

    sampling_rate = 16_000
    max_audio_seconds = 30.0

    def __init__(self, device: str | None = None) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_samples = int(self.max_audio_seconds * self.sampling_rate)

    def _maybe_set_audio_collator(self, inputs: DataLoader[BatchedInput]) -> None:
        inputs.collate_fn = AudioCollator(
            target_sampling_rate=self.sampling_rate,
            max_samples=self.max_audio_samples,
        )

    @torch.inference_mode()
    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding text"):
            embeddings.append(self._encode_text_batch(batch["text"]))
        return torch.cat(embeddings, dim=0).float().cpu()

    @torch.inference_mode()
    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding image"):
            images = [img.convert("RGB") for img in batch["image"]]
            embeddings.append(self._encode_image_batch(images))
        return torch.cat(embeddings, dim=0).float().cpu()

    @torch.inference_mode()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        self._maybe_set_audio_collator(inputs)
        embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding audio"):
            audio_arrays = [audio["array"] for audio in batch["audio"]]
            embeddings.append(self._encode_audio_batch(audio_arrays))
        return torch.cat(embeddings, dim=0).float().cpu()

    @torch.inference_mode()
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
                f"{self.model_name} supports text, image, and/or audio inputs"
            )

        if has_audio:
            self._maybe_set_audio_collator(inputs)

        show_progress_bar = kwargs.get("show_progress_bar", True)
        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            parts: list[torch.Tensor] = []
            if has_text and batch.get("text"):
                parts.append(self._encode_text_batch(batch["text"]))
            if has_image and batch.get("image"):
                images = [img.convert("RGB") for img in batch["image"]]
                parts.append(self._encode_image_batch(images))
            if has_audio and batch.get("audio"):
                audio_arrays = [audio["array"] for audio in batch["audio"]]
                parts.append(self._encode_audio_batch(audio_arrays))
            if not parts:
                raise ValueError(
                    f"No supported modality found in batch: {batch.keys()}"
                )
            fused = parts[0]
            for part in parts[1:]:
                fused += part
            fused = F.normalize(fused, p=2, dim=-1)
            all_embeddings.append(fused)
        return torch.cat(all_embeddings, dim=0).float().cpu()


class SemanticRouterMultiModalEmbedSmallWrapper(SemanticRouterMultiModalEmbedWrapper):
    """Wrapper for llm-semantic-router/multi-modal-embed-small (text/image/audio)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import hf_hub_download

        super().__init__(device=device)
        self.model_name = model_name

        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename="model.pt",
            revision=revision,
        )
        self.model = _MultiModalEmbedSmall()
        self.model.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode_text(texts)

    def _encode_image_batch(self, images: list[Any]) -> torch.Tensor:
        return self.model.encode_image(images)

    def _encode_audio_batch(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        return self.model.encode_audio(waveforms)


class SemanticRouterMultiModalEmbedLargeWrapper(SemanticRouterMultiModalEmbedWrapper):
    """Wrapper for llm-semantic-router/multi-modal-embed-large (text/image/audio)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import hf_hub_download

        super().__init__(device=device)
        self.model_name = model_name

        config_path = hf_hub_download(
            repo_id=model_name, filename="config.json", revision=revision
        )
        checkpoint_path = hf_hub_download(
            repo_id=model_name, filename="model.pt", revision=revision
        )
        with Path(config_path).open(encoding="utf-8") as handle:
            cfg = json.load(handle)
        model_cfg = cfg.get("model", cfg)

        self.model = _MultiModalEmbedLarge(
            text_encoder_name=model_cfg["text_encoder_name"],
            image_encoder_name=model_cfg["image_encoder_name"],
            audio_encoder_name=model_cfg["audio_encoder_name"],
            embedding_dim=int(model_cfg["embedding_dim"]),
            max_text_length=int(model_cfg["max_text_length"]),
        )
        self.model.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode_text(texts)

    def _encode_image_batch(self, images: list[Any]) -> torch.Tensor:
        return self.model.encode_image(images)

    def _encode_audio_batch(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        return self.model.encode_audio(waveforms)


multi_modal_embed_small = ModelMeta(
    loader=SemanticRouterMultiModalEmbedSmallWrapper,
    name="llm-semantic-router/multi-modal-embed-small",
    revision="fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4",
    release_date="2026-02-05",
    languages=["eng-Latn"],
    n_parameters=264_265_730,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=1008,
    max_tokens=256,
    embed_dim=[32, 64, 128, 256, 384],
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/llm-semantic-router/multi-modal-embed-small",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={
        # LLaVA-CC3M, COCO Captions (not in MTEB)
        "LibriSpeech",
    },
    modalities=["text", "image", "audio"],
    model_type=["dense"],
    citation=_CITATION,
)

multi_modal_embed_large = ModelMeta(
    loader=SemanticRouterMultiModalEmbedLargeWrapper,
    name="llm-semantic-router/multi-modal-embed-large",
    revision="e21cde3ccc414c56f504b322662f42c603a939ee",
    release_date="2026-05-03",
    languages=bgem3_languages,
    n_parameters=2_206_806_066,
    n_embedding_parameters=196_608_000,
    memory_usage_mb=8418,
    max_tokens=32768,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers", "Transformers"],
    reference="https://huggingface.co/llm-semantic-router/multi-modal-embed-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    adapted_from="llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
    modalities=["text", "image", "audio"],
    model_type=["dense"],
    citation=_CITATION,
)
