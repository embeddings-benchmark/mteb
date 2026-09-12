#!/usr/bin/env python3
"""Measure how audio sampling budget affects retrieval scores and model rankings.

For each budget (max seconds of audio per clip), encodes the a2i_h and a2i_s
queries at that truncation length, computes ndcg@10 and Spearman rank
correlation against the full-budget baseline. Reports speedup factor per budget.

Addresses MTEB issue #5362.

Examples:
  uv run python scripts/analysis/sampling_budget.py \\
      --model jinaai/jina-embeddings-v5-omni-nano \\
      --model jinaai/jina-embeddings-v5-omni-small \\
      --out results/sampling_budget.json
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
from datasets import Audio, load_dataset
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

import mteb
from mteb.types import PromptType

warnings.filterwarnings("ignore", category=UserWarning, module="mteb")

_REPO = "rakshi719/coco-modality-equivalence"
_TARGET_SR = 16_000

# Audio query configs to test (corpus is always images)
_AUDIO_CONFIGS = [
    ("a2i_h", "COCOModalEquivA2IHumanRetrieval"),
    ("a2i_s", "COCOModalEquivA2ITTSRetrieval"),
]

# Budgets in seconds; None = full clip (no truncation)
_BUDGETS_S: list[float | None] = [1.0, 2.0, 3.0, 5.0, None]


# ── audio helpers ─────────────────────────────────────────────────────────────


def _decode_audio(
    raw: bytes, target_sr: int = _TARGET_SR, max_seconds: float | None = None
) -> dict:
    import torchaudio

    array, sr = sf.read(io.BytesIO(raw))
    if array.ndim > 1:
        array = array.mean(axis=-1)
    array = array.astype(np.float32)
    if sr != target_sr:
        t = torch.from_numpy(array).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        array = t.squeeze(0).numpy()
    if max_seconds is not None:
        max_samples = int(max_seconds * target_sr)
        array = array[:max_samples]
    return {"array": array, "sampling_rate": target_sr, "path": None}


class _AudioDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.features = {"audio": {}, "id": None}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


def _make_audio_loader(
    ds: Any, batch_size: int, max_seconds: float | None
) -> DataLoader:
    ds_raw = ds.cast_column("audio", Audio(decode=False))
    rows = [
        {
            "audio": _decode_audio(r["audio"]["bytes"], max_seconds=max_seconds),
            "id": str(r["id"]),
        }
        for r in ds_raw
    ]
    return DataLoader(_AudioDataset(rows), batch_size=batch_size, shuffle=False)


def _audio_durations(ds: Any) -> np.ndarray:
    """Return array of clip durations in seconds from raw bytes."""
    ds_raw = ds.cast_column("audio", Audio(decode=False))
    durs = []
    for r in ds_raw:
        arr, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        durs.append(len(arr) / sr)
    return np.array(durs)


# ── encoding & retrieval ──────────────────────────────────────────────────────


def _encode(model: Any, loader: DataLoader, task: Any, prompt_type: str) -> np.ndarray:
    pt = PromptType(prompt_type)
    embs = model.encode(
        loader,
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=pt,
    )
    if isinstance(embs, torch.Tensor):
        embs = embs.cpu().numpy()
    return np.array(embs, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def _ndcg_at_k(
    sims: np.ndarray,
    q_ids: list[str],
    c_ids: list[str],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    scores = []
    for qi, qid in enumerate(q_ids):
        gold = qrels.get(qid, {})
        if not gold:
            continue
        ranked = np.argsort(-sims[qi])[:k]
        dcg = sum(
            gold.get(c_ids[ci], 0) / np.log2(r + 2) for r, ci in enumerate(ranked)
        )
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _load_qrels(qrels_ds: Any) -> dict[str, dict[str, int]]:
    import re

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = re.sub(r"^[a-z]+_", "", str(row["query-id"]))
        cid = str(row["corpus-id"])
        qrels.setdefault(qid, {})[cid] = int(row["score"])
    return qrels


# ── per-config budget sweep ───────────────────────────────────────────────────


def _sweep_config(
    model: Any,
    cfg: str,
    task_name: str,
    batch_size: int,
    budgets: list[float | None],
) -> dict:
    task = mteb.get_task(task_name)

    corpus_ds = load_dataset(_REPO, f"{cfg}-corpus", split="test")
    queries_ds = load_dataset(_REPO, f"{cfg}-queries", split="test")
    qrels_ds = load_dataset(_REPO, f"{cfg}-qrels", split="test")

    q_ids = [str(i) for i in queries_ds["id"]]
    c_ids = [str(i) for i in corpus_ds["id"]]
    qrels = _load_qrels(qrels_ds)

    # Measure actual clip durations for speedup calculation
    durations = _audio_durations(queries_ds)
    avg_dur = float(durations.mean())
    print(f"    [{cfg}] avg clip duration: {avg_dur:.1f}s")

    # Encode corpus once (images — no truncation needed)
    from mteb._create_dataloaders import create_dataloader

    print(f"    [{cfg}] encoding corpus…", end=" ", flush=True)
    c_loader = create_dataloader(
        corpus_ds,
        task_metadata=task.metadata,
        prompt_type="document",
        batch_size=batch_size,
    )
    c_embs = _encode(model, c_loader, task, "document")
    print(f"shape={c_embs.shape}")

    # Full-budget baseline first
    full_sims: np.ndarray | None = None
    budget_results: list[dict] = []

    for budget in budgets:
        label = f"{budget}s" if budget is not None else "full"
        effective = min(budget, avg_dur) if budget is not None else avg_dur
        speedup = avg_dur / effective

        print(
            f"    [{cfg}] budget={label}  speedup≈{speedup:.1f}x  encoding…",
            end=" ",
            flush=True,
        )
        q_loader = _make_audio_loader(queries_ds, batch_size, budget)
        q_embs = _encode(model, q_loader, task, "query")
        print(f"shape={q_embs.shape}", end=" ")

        sims = _cosine_sim(q_embs, c_embs)
        ndcg = _ndcg_at_k(sims, q_ids, c_ids, qrels)
        print(f"ndcg@10={ndcg:.4f}")

        if budget is None:
            full_sims = sims

        budget_results.append(
            {
                "budget_s": budget,
                "label": label,
                "speedup": round(speedup, 2),
                "ndcg_at_10": round(ndcg, 4),
                "sims": sims,  # kept temporarily for Spearman
            }
        )

    # Compute Spearman correlation of per-query similarity vectors vs full budget
    assert full_sims is not None
    for r in budget_results:
        sims_b = r.pop("sims")
        # For each query, Spearman between full-budget and budget sim row
        query_rhos = [
            spearmanr(full_sims[qi], sims_b[qi]).statistic for qi in range(len(q_ids))
        ]
        r["spearman_vs_full"] = round(float(np.mean(query_rhos)), 4)

    print(f"\n    [{cfg}] Budget summary:")
    for r in budget_results:
        print(
            f"      {r['label']:>5}  speedup={r['speedup']:.1f}x"
            f"  ndcg@10={r['ndcg_at_10']:.4f}"
            f"  spearman={r['spearman_vs_full']:.4f}"
        )

    return {
        "config": cfg,
        "avg_clip_duration_s": round(avg_dur, 2),
        "budgets": budget_results,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def compute_budget(model_name: str, batch_size: int, device: str | None) -> dict:
    print(f"\n=== {model_name} ===")
    model = mteb.get_model(model_name, device=device)

    config_results = []
    for cfg, task_name in _AUDIO_CONFIGS:
        config_results.append(
            _sweep_config(model, cfg, task_name, batch_size, _BUDGETS_S)
        )

    return {"model": model_name, "configs": config_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        results.append(compute_budget(model_name, args.batch_size, args.device))

    output = json.dumps(results, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"\nResults written to {out_path}")
    else:
        print("\n=== SUMMARY ===")
        print(output)


if __name__ == "__main__":
    main()
