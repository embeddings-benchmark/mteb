#!/usr/bin/env python3
"""Compute modality gap for omni-embedding models.

Loads the COCOModalEquiv dataset (120 MSCOCO items each available as image,
text caption, human-spoken audio and TTS audio), encodes each modality with
the requested model(s), then reports:

  * pairwise centroid cosine-distance between modalities (the "gap")
  * intra-modal average pairwise cosine-distance (spread)

Follows the definition in Liang et al. NeurIPS 2022 and the motivation in
MTEB issue #5359.

Examples:
  uv run python scripts/analysis/modality_gap.py \\
      --model Tevatron/OmniEmbed-v0.1
  uv run python scripts/analysis/modality_gap.py \\
      --model Tevatron/OmniEmbed-v0.1 \\
      --model nvidia/omni-embed-nemotron-3b \\
      --out results/modality_gap.json
"""

from __future__ import annotations

import argparse
import io
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Image, load_dataset
from torch.utils.data import DataLoader

import mteb
from mteb._create_dataloaders import create_dataloader
from mteb.types import PromptType

warnings.filterwarnings("ignore", category=UserWarning, module="mteb")

_REPO = "rakshi719/coco-modality-equivalence"

# (modality_label, hf_config, hf_split_type, task_name, prompt_type)
_MODALITIES: list[tuple[str, str, str, str, str]] = [
    ("text", "t2i", "queries", "COCOModalEquivT2IRetrieval", "query"),
    ("image", "t2i", "corpus", "COCOModalEquivT2IRetrieval", "document"),
    ("audio_human", "a2i_h", "queries", "COCOModalEquivA2IHumanRetrieval", "query"),
    ("audio_tts", "a2i_s", "queries", "COCOModalEquivA2ITTSRetrieval", "query"),
]

_TARGET_SR = 16_000


def _load_ds(config_prefix: str, split_type: str) -> Any:
    """Load one HF config split, e.g. 't2i-corpus'."""
    return load_dataset(_REPO, f"{config_prefix}-{split_type}", split="test")


def _decode_audio_bytes(raw: bytes, target_sr: int = _TARGET_SR) -> dict:
    """Decode raw audio bytes to {'array': np.ndarray, 'sampling_rate': int}."""
    import torchaudio

    array, sr = sf.read(io.BytesIO(raw))
    if array.ndim > 1:
        array = array.mean(axis=-1)
    array = array.astype(np.float32)
    if sr != target_sr:
        t = torch.from_numpy(array).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        array = t.squeeze(0).numpy()
    return {"array": array, "sampling_rate": target_sr, "path": None}


class _AudioDataset:
    """Minimal dataset whose .features satisfy the MTEB AudioCollator detection."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        # Value of each key doesn't matter — only the presence of "audio" is checked.
        self.features = {"audio": {}, "id": None}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


def _make_audio_loader(ds: Any, batch_size: int) -> DataLoader:
    """Build a DataLoader for audio that decodes bytes via soundfile (no torchcodec)."""
    ds_raw = ds.cast_column("audio", Audio(decode=False))
    rows = [
        {"audio": _decode_audio_bytes(r["audio"]["bytes"]), "id": r["id"]}
        for r in ds_raw
    ]
    return DataLoader(_AudioDataset(rows), batch_size=batch_size, shuffle=False)


def _encode_all(
    model: Any,
    loader: DataLoader,
    task: Any,
    prompt_type: str | None,
) -> np.ndarray:
    """Encode a full dataloader and return (N, D) float32 array."""
    pt = PromptType(prompt_type) if prompt_type else None
    arrays = model.encode(
        loader,
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=pt,
    )
    if isinstance(arrays, torch.Tensor):
        arrays = arrays.cpu().numpy()
    return np.array(arrays, dtype=np.float32)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors: 1 - cos_sim."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


def _intra_spread(embs: np.ndarray, n_sample: int = 500) -> float:
    """Average pairwise cosine distance within a modality (sampled)."""
    n = len(embs)
    idx = np.random.choice(n, size=min(n_sample, n), replace=False)
    sub = embs[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    sims = sub @ sub.T
    mask = ~np.eye(len(sub), dtype=bool)
    return float(1.0 - sims[mask].mean())


def compute_gap(model_name: str, batch_size: int, device: str | None) -> dict:
    print(f"\n=== {model_name} ===")
    model = mteb.get_model(model_name, device=device)

    embeddings: dict[str, np.ndarray] = {}

    for modality_label, cfg_prefix, split_type, task_name, prompt_type in _MODALITIES:
        ds = _load_ds(cfg_prefix, split_type)
        task = mteb.get_task(task_name)

        is_audio = modality_label.startswith("audio")
        if is_audio:
            loader = _make_audio_loader(ds, batch_size)
        else:
            loader = create_dataloader(
                ds,
                task_metadata=task.metadata,
                prompt_type=prompt_type,
                batch_size=batch_size,
            )

        print(f"  Encoding {modality_label} ({len(ds)} items)…", end=" ", flush=True)
        try:
            embs = _encode_all(model, loader, task, prompt_type)
            print(f"shape={embs.shape}")
            embeddings[modality_label] = embs
        except Exception as e:
            print(f"FAILED: {e}")

    if len(embeddings) < 2:
        return {"model": model_name, "error": "fewer than 2 modalities encoded"}

    centroids = {m: embs.mean(axis=0) for m, embs in embeddings.items()}

    modality_keys = list(embeddings.keys())
    gap_matrix: dict[str, float] = {}
    for i, ma in enumerate(modality_keys):
        for mb in modality_keys[i + 1 :]:
            gap_matrix[f"{ma}_vs_{mb}"] = _cosine_distance(centroids[ma], centroids[mb])

    spread: dict[str, float] = {
        m: _intra_spread(embs) for m, embs in embeddings.items()
    }

    print("  Gap (centroid cosine distances):")
    for k, v in gap_matrix.items():
        print(f"    {k}: {v:.4f}")
    print("  Intra-modal spread:")
    for k, v in spread.items():
        print(f"    {k}: {v:.4f}")

    return {
        "model": model_name,
        "n_items": {m: len(embeddings[m]) for m in embeddings},
        "gap": gap_matrix,
        "spread": spread,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model name (repeat for multiple models)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 'cuda', 'cpu'")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    results = []
    for model_name in args.models:
        result = compute_gap(model_name, args.batch_size, args.device)
        results.append(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out_path}")
    else:
        print("\n=== SUMMARY ===")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
