from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.models.model_implementations.qwen3_vl_embedding_models import (
    Qwen3VLEmbeddingWrapper,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

VL_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
VL_REVISION = "2c4565515e0f265c6511776e7193b22c0968ddc7"
OMNI_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"

# The audio branch was trained against the first 3584 dims of the 4096-dim
# Qwen3-VL-Embedding output, so the visual branch must be truncated to match.
EMBED_DIM = 3584
AUDIO_TOWER_DIM = 3584
SAMPLING_RATE = 16000

_MODALITY_COLUMNS = ("text", "image", "audio", "video")


class _LightweightAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = dim // reduction
        self.down = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class _ModalityAdapterFusion(nn.Module):
    """Four adapters combined by a softmax gate computed from the input itself."""

    target_modalities = ("text", "audio", "video", "label")

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {m: _LightweightAdapter(dim, reduction) for m in self.target_modalities}
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, len(self.target_modalities)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(x), dim=-1)
        stacked = torch.stack(
            [self.adapters[m](x) for m in self.target_modalities], dim=1
        )
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)


class _TianmuAudioBranch(nn.Module):
    """Qwen2.5-Omni audio tower plus the trained Tianmu connector/adapter stack."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        from transformers import AutoConfig
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoder,
        )

        super().__init__()
        config = AutoConfig.from_pretrained(OMNI_MODEL_NAME)
        audio_config = getattr(config, "thinker_config", config).audio_config
        self.encoder = Qwen2_5OmniAudioEncoder(audio_config)
        self.connector = nn.Sequential(
            nn.Linear(AUDIO_TOWER_DIM, 2 * AUDIO_TOWER_DIM, bias=False),
            nn.LayerNorm(2 * AUDIO_TOWER_DIM),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(2 * AUDIO_TOWER_DIM, 4 * AUDIO_TOWER_DIM, bias=False),
            nn.LayerNorm(4 * AUDIO_TOWER_DIM),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(4 * AUDIO_TOWER_DIM, dim, bias=False),
        )
        self.adapter_fusion = _ModalityAdapterFusion(dim)
        self.projection = nn.RMSNorm(dim, eps=1e-6)

    def load_tianmu_weights(self, model_name: str, revision: str | None) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        state: dict[str, torch.Tensor] = {}
        for filename in (
            "tianmu_audio_core.safetensors",
            "Prototype-adapter/model.safetensors",
        ):
            path = hf_hub_download(model_name, filename, revision=revision)
            state.update(load_file(path, device="cpu"))

        remapped: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key.startswith("audio_encoder."):
                remapped[key[len("audio_encoder.") :]] = tensor
            elif key.startswith("audio_connector.mlp."):
                remapped["connector." + key[len("audio_connector.mlp.") :]] = tensor
            elif key.startswith("projection.norm."):
                remapped["projection." + key[len("projection.norm.") :]] = tensor
            elif key.startswith("adapter_fusion."):
                remapped[key] = tensor
            # prototypes.* are training-loss only and unused at inference

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected Tianmu audio weights: {sorted(unexpected)[:5]}"
            )
        if missing:
            logger.warning("Tianmu audio branch: %d missing keys", len(missing))

    @torch.no_grad()
    def forward(
        self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                input_features.shape[0], input_features.shape[-1], dtype=torch.long
            )
        pooled = []
        # The Omni audio tower takes one packed 2D sample at a time; the reference
        # implementation encodes per sample rather than using the packed protocol.
        for raw_feature, mask in zip(
            input_features, feature_attention_mask, strict=True
        ):
            mel = raw_feature.to(device)
            length = min(max(int(mask.sum().item()), 1), mel.shape[-1])
            if length < 4:
                pooled.append(torch.zeros(AUDIO_TOWER_DIM, device=device))
                continue
            mel = mel[:, :length].contiguous()
            feature_lens = torch.tensor([length], dtype=torch.long, device=device)
            aftercnn_lens = ((feature_lens - 1) // 2 + 1).clamp(min=1)
            output = self.encoder(
                mel, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens
            )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output[0] if isinstance(output, tuple) else output
            pooled.append(hidden.mean(dim=-2).reshape(-1))

        embeddings = torch.stack(pooled, dim=0)
        embeddings = self.connector(embeddings)
        embeddings = self.adapter_fusion(embeddings)
        embeddings = self.projection(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)


class TianmuEmbUniWrapper(Qwen3VLEmbeddingWrapper):
    """Frozen Qwen3-VL-Embedding-8B for text/image/video plus a trained audio branch.

    Text, image and video (including their combinations) are fused natively by the
    Qwen3-VL backbone in a single forward pass. Audio has no native fusion path, so
    when it co-occurs with another modality the per-branch unit vectors are summed,
    following the convention used by the CLIP and CLAP wrappers.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_seconds: float = 30.0,
        embed_dim: int = EMBED_DIM,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoFeatureExtractor

        if embed_dim != EMBED_DIM:
            raise ValueError(
                f"Tianmu-Emb-Uni only supports embed_dim={EMBED_DIM}; the audio branch "
                f"was aligned against that space and cannot be truncated further."
            )

        # Truncation to EMBED_DIM is applied in _to_shared_space rather than by the
        # SentenceTransformer, which would normalize the 4096-dim vector first.
        super().__init__(
            VL_MODEL_NAME,
            revision=VL_REVISION,
            device=device,
            **kwargs,
        )
        self.model_name = model_name
        self.max_audio_samples = int(max_audio_seconds * SAMPLING_RATE)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(OMNI_MODEL_NAME)

        self.audio_branch = _TianmuAudioBranch()
        self.audio_branch.load_tianmu_weights(model_name, revision)
        self.audio_branch.to(self.model.device).eval()

    @staticmethod
    def _subset_loader(
        inputs: DataLoader[BatchedInput], keep: set[str]
    ) -> DataLoader[BatchedInput]:
        dataset = inputs.dataset
        drop = [
            column
            for column in _MODALITY_COLUMNS
            if column in dataset.column_names and column not in keep
        ]
        if not drop:
            return inputs
        return DataLoader(
            dataset.remove_columns(drop),
            batch_size=inputs.batch_size,
            collate_fn=inputs.collate_fn,
            num_workers=inputs.num_workers,
            shuffle=False,
        )

    @torch.no_grad()
    def _encode_audio(self, inputs: DataLoader[BatchedInput]) -> np.ndarray:
        from mteb.models.modality_collators import AudioCollator

        inputs.collate_fn = AudioCollator(target_sampling_rate=SAMPLING_RATE)
        embeddings = []
        for batch in tqdm(inputs, desc="Encoding audio"):
            arrays = [
                np.asarray(item["array"], dtype=np.float32) for item in batch["audio"]
            ]
            features = self.feature_extractor(
                arrays,
                sampling_rate=SAMPLING_RATE,
                return_attention_mask=True,
                padding="max_length",
                max_length=self.max_audio_samples,
                truncation=True,
                return_tensors="pt",
            )
            mask = features.get(
                "attention_mask", features.get("feature_attention_mask")
            )
            embeddings.append(
                self.audio_branch(features["input_features"], mask)
                .cpu()
                .float()
                .numpy()
            )
        return np.concatenate(embeddings, axis=0)

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
        features = set(inputs.dataset.features)
        vl_modalities = {m for m in ("text", "image", "video") if m in features}

        if "audio" not in features:
            return _to_shared_space(
                super().encode(
                    inputs,
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    prompt_type=prompt_type,
                    **kwargs,
                )
            )

        audio_embeddings = self._encode_audio(self._subset_loader(inputs, {"audio"}))
        if not vl_modalities:
            return audio_embeddings

        vl_embeddings = _to_shared_space(
            super().encode(
                self._subset_loader(inputs, vl_modalities),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )
        )
        return vl_embeddings + audio_embeddings


def _to_shared_space(embeddings: Array) -> np.ndarray:
    """Cut the 4096-dim Qwen3-VL output down to the space the audio branch was aligned to."""
    array = (
        embeddings.detach().cpu().numpy()
        if isinstance(embeddings, torch.Tensor)
        else np.asarray(embeddings)
    )
    array = array.astype(np.float32, copy=False)[..., :EMBED_DIM]
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


TIANMU_CITATION = """@misc{tianmuembuni2026,
  title={Tianmu-Emb-Uni-8B: A Unified Multimodal Embedding Model},
  author={TianmuLab},
  year={2026},
  url={https://huggingface.co/TianmuLab/Tianmu-Emb-Uni}
}"""

tianmu_emb_uni = ModelMeta(
    loader=TianmuEmbUniWrapper,
    name="TianmuLab/Tianmu-Emb-Uni",
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="89b33e5c6c13fe12b49646e13dfb344334c7ace5",
    release_date="2026-08-05",
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    n_parameters=8_980_609_525,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=34_251,
    embed_dim=EMBED_DIM,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/TianmuLab/Tianmu-Emb-Uni",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    # Inferred from the released training scripts (configs and dataset parsers in
    # eval/mmeb_v3_eval and tianmu_model/adapter.py); not an official statement.
    training_datasets={
        "ClothoA2TRetrieval",
        "ClothoT2ARetrieval",
        "SpeechCocoA2IRetrieval",
        "SpeechCocoI2ARetrieval",
        "SoundDescsA2TRetrieval",
        "SoundDescsT2ARetrieval",
        "ESC50",
        "ESC50Clustering",
        "CREMADPairClassification",
        "SpeechCommands",
        "NSynth",
        "UrbanSound8k",
        "UrbanSound8KT2ARetrieval",
        "UrbanSound8KA2TRetrieval",
        "AVEDatasetClassification",
    },
    adapted_from="Qwen/Qwen3-VL-Embedding-8B",
    citation=TIANMU_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)
