#!/usr/bin/env python3
"""Measure directional asymmetry in cross-modal retrieval.

For each modality pair (text↔image, audio↔image), computes:
  asymmetry = ndcg@10(A→B) − ndcg@10(B→A)

Uses the COCOModalEquiv tasks which share an identical 120-item MSCOCO pool,
so asymmetry reflects modality rather than content differences.

Addresses MTEB issue #5360.

Examples:
  uv run python scripts/analysis/directional_asymmetry.py \\
      --model jinaai/jina-embeddings-v5-omni-nano \\
      --model jinaai/jina-embeddings-v5-omni-small \\
      --out results/directional_asymmetry.json
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
from torch.utils.data import DataLoader

import mteb
from mteb._create_dataloaders import create_dataloader
from mteb.types import PromptType

warnings.filterwarnings("ignore", category=UserWarning, module="mteb")

_REPO = "rakshi719/coco-modality-equivalence"
_TARGET_SR = 16_000

# (label, forward_config, reverse_config)
# Each config maps to {cfg}-corpus, {cfg}-queries, {cfg}-qrels on HF.
_PAIRS: list[tuple[str, str, str]] = [
    ("text↔image", "t2i", "i2t"),
    ("audio_human↔image", "a2i_h", "i2a_h"),
    ("audio_tts↔image", "a2i_s", "i2a_s"),
]

_CFG_TO_TASK = {
    "t2i": "COCOModalEquivT2IRetrieval",
    "i2t": "COCOModalEquivI2TRetrieval",
    "a2i_h": "COCOModalEquivA2IHumanRetrieval",
    "i2a_h": "COCOModalEquivI2AHumanRetrieval",
    "a2i_s": "COCOModalEquivA2ITTSRetrieval",
    "i2a_s": "COCOModalEquivI2ATTSRetrieval",
}


# ── audio helpers (same approach as modality_gap.py) ─────────────────────────


def _decode_audio_bytes(raw: bytes, target_sr: int = _TARGET_SR) -> dict:
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
    """Minimal dataset whose .features satisfy MTEB's AudioCollator detection."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.features = {"audio": {}, "id": None}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


def _make_audio_loader(ds: Any, batch_size: int) -> DataLoader:
    ds_raw = ds.cast_column("audio", Audio(decode=False))
    rows = [
        {"audio": _decode_audio_bytes(r["audio"]["bytes"]), "id": str(r["id"])}
        for r in ds_raw
    ]
    return DataLoader(_AudioDataset(rows), batch_size=batch_size, shuffle=False)


# ── encoding ──────────────────────────────────────────────────────────────────


def _encode_loader(
    model: Any, loader: DataLoader, task: Any, prompt_type: str
) -> np.ndarray:
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


# ── retrieval metric ──────────────────────────────────────────────────────────


def _ndcg_at_k(
    q_ids: list[str],
    c_ids: list[str],
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    c_norm = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-9)
    sims = q_norm @ c_norm.T  # (N_q, N_c)

    scores = []
    for qi, qid in enumerate(q_ids):
        gold = qrels.get(qid, {})
        if not gold:
            continue
        ranked = np.argsort(-sims[qi])[:k]
        dcg = sum(
            gold.get(c_ids[ci], 0) / np.log2(rank + 2) for rank, ci in enumerate(ranked)
        )
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ── qrel loading ──────────────────────────────────────────────────────────────


def _load_qrels(qrels_ds: Any) -> dict[str, dict[str, int]]:
    """Load qrels, normalizing query-ids that carry h_/s_ prefixes.

    In a2i_h/a2i_s configs the qrels use h_{img_id}/s_{img_id} as query-ids,
    but the queries dataset stores items with plain img_ids. Strip the prefix
    so they match.
    """
    import re

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = int(row["score"])
        # Strip leading letter+underscore prefix (e.g. "h_", "s_")
        qid = re.sub(r"^[a-z]+_", "", qid)
        qrels.setdefault(qid, {})[cid] = score
    return qrels


# ── per-config evaluation ─────────────────────────────────────────────────────


def _eval_config(model: Any, cfg: str, batch_size: int) -> float:
    task = mteb.get_task(_CFG_TO_TASK[cfg])

    corpus_ds = load_dataset(_REPO, f"{cfg}-corpus", split="test")
    queries_ds = load_dataset(_REPO, f"{cfg}-queries", split="test")
    qrels_ds = load_dataset(_REPO, f"{cfg}-qrels", split="test")

    q_ids = [str(i) for i in queries_ds["id"]]
    c_ids = [str(i) for i in corpus_ds["id"]]
    qrels = _load_qrels(qrels_ds)

    def _loader(ds: Any, prompt_type: str) -> DataLoader:
        if "audio" in ds.column_names:
            return _make_audio_loader(ds, batch_size)
        return create_dataloader(
            ds,
            task_metadata=task.metadata,
            prompt_type=prompt_type,
            batch_size=batch_size,
        )

    print(f"    [{cfg}] queries ({len(queries_ds)})…", end=" ", flush=True)
    q_embs = _encode_loader(model, _loader(queries_ds, "query"), task, "query")
    print(f"shape={q_embs.shape}")

    print(f"    [{cfg}] corpus  ({len(corpus_ds)})…", end=" ", flush=True)
    c_embs = _encode_loader(model, _loader(corpus_ds, "document"), task, "document")
    print(f"shape={c_embs.shape}")

    score = _ndcg_at_k(q_ids, c_ids, q_embs, c_embs, qrels)
    print(f"    [{cfg}] ndcg@10 = {score:.4f}")
    return score


# ── main ──────────────────────────────────────────────────────────────────────


def compute_asymmetry(model_name: str, batch_size: int, device: str | None) -> dict:
    print(f"\n=== {model_name} ===")
    model = mteb.get_model(model_name, device=device)

    ndcg: dict[str, float] = {}
    for _, fwd, rev in _PAIRS:
        for cfg in (fwd, rev):
            try:
                ndcg[cfg] = _eval_config(model, cfg, batch_size)
            except Exception as e:
                print(f"    [{cfg}] FAILED: {e}")

    asymmetries: dict[str, dict] = {}
    print("\n  Asymmetries (A→B − B→A):")
    for label, fwd, rev in _PAIRS:
        s_fwd = ndcg.get(fwd)
        s_rev = ndcg.get(rev)
        if s_fwd is not None and s_rev is not None:
            asym = s_fwd - s_rev
            dominant = "A→B" if asym > 0 else ("B→A" if asym < 0 else "symmetric")
            sign = "+" if asym >= 0 else ""
            print(
                f"    {label}: {s_fwd:.4f} − {s_rev:.4f} = {sign}{asym:.4f}  [{dominant}]"
            )
            asymmetries[label] = {
                "forward_config": fwd,
                "reverse_config": rev,
                "forward_score": round(s_fwd, 4),
                "reverse_score": round(s_rev, 4),
                "asymmetry": round(asym, 4),
                "dominant": dominant,
            }
        else:
            asymmetries[label] = {
                "error": f"fwd={s_fwd} rev={s_rev}",
            }
            print(f"    {label}: incomplete")

    return {
        "model": model_name,
        "ndcg_at_10": ndcg,
        "asymmetry": asymmetries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model name (repeat for multiple)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        results.append(compute_asymmetry(model_name, args.batch_size, args.device))

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
