"""MTEB model implementation: google/embeddinggemma-2 + frozen null-space audio head (Gladia).

Drop-in for `mteb/models/model_implementations/` (file name: gladia_embeddinggemma2_nullspace.py).

What it does
------------
* Text, image and video inputs: identical code path to `EmbeddingGemma2Wrapper` (the base class
  `encode` is called unchanged, no hook is active, no extra tensor operation happens).
* Audio inputs: the base embedding v (768d) is computed exactly as in `EmbeddingGemma2Wrapper`, while a
  forward hook on `audio_tower` records the frozen-PCA intermediates of every clip:
      L2_ms   = masked mean+std of Conformer layer 2 (2048d, unit-normalised)
      L6_mean = masked mean of Conformer layer 6     (1024d, unit-normalised)
  Then, with the frozen inductive ext_soft_whiten PCA states (fit on external corpora only):
      z_1 = normalize( ((L2_ms - mean_1) @ proj_1) / (scale_1 + eps) )      (128d)
      z_2 = normalize( ((L6m   - mean_2) @ proj_2) / (scale_2 + eps) )      (128d)
      z   = normalize( cat[ sqrt(.5) z_1, sqrt(.5) z_2 ] )                  (256d)
      v_final = normalize( v + gamma * (z @ U_perp^T) )                      (768d, U_perp in R^{768x256})

Only the frozen tensors in `nullspace_audio_head.safetensors` are needed at inference time.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mteb.models.model_implementations.google_embeddinggemma import (
    EMBEDDING_GEMMA_CITATION,
    MULTILINGUAL_EVALUATED_LANGUAGES,
    EmbeddingGemma2Wrapper,
)
from mteb.models.model_implementations.google_gemini import GECKO_TRAINING_DATA
from mteb.models.model_meta import ModelMeta

BASE_MODEL_ID = "google/embeddinggemma-2"
BASE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"

HEAD_REPO_ID = "BrunoHays/embeddinggemma-2-nullspace-audio"
HEAD_FILENAME = "nullspace_audio_head.safetensors"
# TODO(before PR): pin to the commit hash returned by `hf upload` instead of the moving branch.
HEAD_REVISION = "0222035d2bb6e8baa9949b026a87368c520f8eba"

EPS = 1e-5
HEAD_DIM = 256
BLOCK_WEIGHT = math.sqrt(128 / HEAD_DIM)  # sqrt(0.5): each 128d block carries half the energy

# Datasets used ONLY to fit the frozen PCA states (Protocol A: no MAEB / MSEB audio involved).
PCA_FIT_DATASETS = {
    "ylacombe/expresso",
    "agkphysics/AudioSet",
    "lewtun/music_genres",
    "ashraq/esc50",
}


# --------------------------------------------------------------------------- pooling (same math as the sweep)


def _frame_mask_to_bool(frame_mask: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_mask.dim() != 2 or frame_mask.shape != hidden.shape[:2]:
        raise ValueError(f"frame_mask {tuple(frame_mask.shape)} must match hidden[:2] {tuple(hidden.shape[:2])}")
    mask = frame_mask.to(device=hidden.device).bool()
    n = mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
    return mask, n


def masked_mean_pool(hidden: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Masked temporal mean, L2-normalised. [B, T, D] -> [B, D]."""
    x = hidden.to(torch.float32)
    mask, n = _frame_mask_to_bool(frame_mask, x)
    x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
    return F.normalize(x.sum(dim=1) / n, p=2, dim=-1)


def masked_mean_std_pool(hidden: torch.Tensor, frame_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Masked temporal [mean, std] concatenated, L2-normalised. [B, T, D] -> [B, 2D]."""
    x = hidden.to(torch.float32)
    mask, n = _frame_mask_to_bool(frame_mask, x)
    m3 = mask.unsqueeze(-1)
    zeros = torch.zeros_like(x)
    mu = torch.where(m3, x, zeros).sum(dim=1, keepdim=True) / n.unsqueeze(-1)  # [B, 1, D]
    diff = torch.where(m3, x - mu, zeros)
    sigma = torch.sqrt((diff**2).sum(dim=1) / n + eps)  # [B, D]
    return F.normalize(torch.cat([mu.squeeze(1), sigma], dim=-1), p=2, dim=-1)


@contextlib.contextmanager
def _clip_order_preserved(st_model: Any):
    """Make SentenceTransformer's length-sort the identity, so hook captures line up with dataset order.

    ST sorts each call by `_input_length` (longest first) and un-sorts at the end. A monotonically
    increasing counter (negated inside the sort) yields the identity permutation. Restored on exit.
    """
    counter = itertools.count()
    st_model._input_length = lambda sample: -next(counter)  # noqa: ARG005
    try:
        yield
    finally:
        st_model.__dict__.pop("_input_length", None)


def _has_audio_column(inputs: Any) -> bool:
    cols = getattr(getattr(inputs, "dataset", None), "column_names", None) or []
    return "audio" in cols


def _load_head(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    t = {k: v.to(torch.float32) for k, v in load_file(str(path)).items()}
    expected = {
        "u_perp": (768, 256),
        "l2_ms_mean": (1, 2048),
        "l2_ms_proj": (2048, 128),
        "l2_ms_scale": (128,),
        "l6_m_mean": (1, 1024),
        "l6_m_proj": (1024, 128),
        "l6_m_scale": (128,),
    }
    for k, shape in expected.items():
        if tuple(t[k].shape) != shape:
            raise ValueError(f"{k}: expected {shape}, got {tuple(t[k].shape)}")
    return t


class EmbeddingGemma2NullSpaceWrapper(EmbeddingGemma2Wrapper):
    """`EmbeddingGemma2Wrapper` + frozen null-space PCA head applied to audio inputs only.

    `model_name` is the submission's ModelMeta name (the loader is called as `loader(meta.name, revision=...)`);
    the checkpoint that is always loaded is `BASE_MODEL_ID` at `revision` (= BASE_REVISION).

    Args:
        gamma: null-space injection strength (1.0 = headline, 0.7 = second submitted point).
        head_path: local safetensors file. If None, downloaded from `HEAD_REPO_ID` at `head_revision`.
        head_revision: revision of the head repo.
        use_image: forwarded to EmbeddingGemma2Wrapper (False for MAEB).
        use_audio: must be True for this submission (the head only acts on audio).
    """

    def __init__(
        self,
        model_name: str = BASE_MODEL_ID,  # noqa: ARG002 - submission name, see docstring
        revision: str | None = BASE_REVISION,
        *,
        gamma: float = 1.0,
        head_path: str | Path | None = None,
        head_revision: str = HEAD_REVISION,
        use_image: bool = False,
        use_audio: bool = True,
        **kwargs: Any,
    ) -> None:
        if not use_audio:
            raise ValueError("EmbeddingGemma2NullSpaceWrapper requires use_audio=True")
        super().__init__(
            BASE_MODEL_ID,
            revision=revision,
            use_image=use_image,
            use_audio=True,
            **kwargs,
        )
        self.gamma = float(gamma)
        if head_path is None:
            from huggingface_hub import hf_hub_download

            head_path = hf_hub_download(HEAD_REPO_ID, HEAD_FILENAME, revision=head_revision)
        self._head = _load_head(head_path)
        self._captures: list[dict[str, torch.Tensor]] = []
        self._recording = False

        # Same audio-tower configuration as the phase-1 runs: hidden_states[k] = Conformer layer k.
        tower = self.model[0].auto_model.audio_tower
        tower.config.tie_last_hidden_states = False
        orig_forward = tower.forward

        def forward_with_hidden(*args: Any, **kw: Any) -> Any:
            kw["output_hidden_states"] = True
            out = orig_forward(*args, **kw)
            if self._recording:
                self._captures.append(self._pool_clips(out))
            return out

        tower.forward = forward_with_hidden

    # ------------------------------------------------------------------ head
    @torch.no_grad()
    def _pool_clips(self, out: Any) -> dict[str, torch.Tensor]:
        mask = out.attention_mask
        if mask is None:
            raise RuntimeError("audio_tower returned no attention_mask; refusing to pool unmasked")
        hs = out.hidden_states
        if hs is None or len(hs) < 7:
            raise RuntimeError("audio_tower did not return hidden_states")
        return {
            "L2_ms": masked_mean_std_pool(hs[2], mask).to("cpu", torch.float32),
            "L6_mean": masked_mean_pool(hs[6], mask).to("cpu", torch.float32),
        }

    @torch.no_grad()
    def null_space_z(self, l2_ms: torch.Tensor, l6_mean: torch.Tensor) -> torch.Tensor:
        """Frozen 256d null-space coordinates for pooled clip features [N, 2048] and [N, 1024]."""
        h = self._head
        z1 = F.normalize(((l2_ms - h["l2_ms_mean"]) @ h["l2_ms_proj"]) / (h["l2_ms_scale"] + EPS), p=2, dim=-1)
        z2 = F.normalize(((l6_mean - h["l6_m_mean"]) @ h["l6_m_proj"]) / (h["l6_m_scale"] + EPS), p=2, dim=-1)
        z = torch.cat([BLOCK_WEIGHT * z1, BLOCK_WEIGHT * z2], dim=-1)
        return F.normalize(z, p=2, dim=-1)

    @torch.no_grad()
    def inject(self, v_base: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """v_final = normalize(v_base + gamma * z @ U_perp^T); v_base is normalised first (as in the sweep)."""
        v = F.normalize(v_base.to(torch.float32), p=2, dim=-1)
        if self.gamma == 0.0:
            return v
        return F.normalize(v + self.gamma * (z @ self._head["u_perp"].T), p=2, dim=-1)

    # ------------------------------------------------------------------ mteb entrypoint
    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        base_kwargs = dict(task_metadata=task_metadata, hf_split=hf_split, hf_subset=hf_subset, prompt_type=prompt_type)
        if not _has_audio_column(inputs):
            # text / image / video: bit-for-bit the EmbeddingGemma2Wrapper path
            return super().encode(inputs, **base_kwargs, **kwargs)

        n_rows = len(inputs.dataset)
        self._captures = []
        self._recording = True
        try:
            with _clip_order_preserved(self.model):
                emb = super().encode(inputs, **base_kwargs, **kwargs)
        finally:
            self._recording = False
            caps, self._captures = self._captures, []

        if len(emb) != n_rows:
            raise RuntimeError(f"encoded {len(emb)} rows for {n_rows} inputs")
        if not caps:
            raise RuntimeError("audio column present but audio_tower never ran")
        l2 = torch.cat([c["L2_ms"] for c in caps], dim=0)
        l6 = torch.cat([c["L6_mean"] for c in caps], dim=0)
        if l2.shape[0] != n_rows:
            raise RuntimeError(f"captured {l2.shape[0]} audio clips for {n_rows} rows; mixed-modality call?")

        v = torch.as_tensor(np.asarray(emb, dtype=np.float32))
        out = self.inject(v, self.null_space_z(l2, l6))
        return out.numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------- model metadata

_COMMON_META: dict[str, Any] = dict(
    loader=EmbeddingGemma2NullSpaceWrapper,  # type: ignore[call-arg]
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision=BASE_REVISION,
    release_date="2026-10-24",
    # Text + audio towers loaded (EmbeddingGemma2Wrapper overwrites these from the loaded towers anyway).
    n_parameters=576_613_664,
    n_embedding_parameters=134_217_728,
    memory_usage_mb=2200,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference=f"https://huggingface.co/{HEAD_REPO_ID}",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(GECKO_TRAINING_DATA) | PCA_FIT_DATASETS,
    similarity_fn_name="cosine",
    modalities=["text", "audio"],
    adapted_from=BASE_MODEL_ID,
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)

gladia_embeddinggemma_2_nullspace_audio = ModelMeta(
    name="BrunoHays/embeddinggemma-2-nullspace-audio",
    loader_kwargs={"gamma": 1.0, "head_revision": HEAD_REVISION},
    **_COMMON_META,
)

gladia_embeddinggemma_2_nullspace_audio_g07 = ModelMeta(
    name="BrunoHays/embeddinggemma-2-nullspace-audio-g0.7",
    loader_kwargs={"gamma": 0.7, "head_revision": HEAD_REVISION},
    **_COMMON_META,
)
