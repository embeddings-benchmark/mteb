#!/usr/bin/env python3
"""Evaluate all FIVR regimes while encoding their shared corpus only once.

The three tasks have identical corpus media but different query instructions and
qrels. This runner checkpoints corpus embedding shards, encodes each query view
with its own task prompt, and uses MTEB's retrieval scorer unchanged.

Example:
    MTEB_FIVR_VIDEO_DIR=/path/to/fivr/videos python \
        scripts/data/fivr/run_shared_evaluation.py \
        --model Qwen/Qwen3-VL-Embedding-2B --device mps \
        --num-frames 16 --batch-size 2 --output /tmp/fivr-qwen.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mteb
import numpy as np
import torch

from mteb._create_dataloaders import create_dataloader
from mteb._evaluators.retrieval_metrics import (
    calculate_retrieval_scores,
    make_score_dict,
)
from mteb.tasks.retrieval.zxx.fivr_5k_retrieval import (
    FIVR5KCSVRRetrieval,
    FIVR5KDSVRRetrieval,
    FIVR5KISVRRetrieval,
)
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset

    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.models import EncoderProtocol


TASK_CLASSES = (
    FIVR5KDSVRRetrieval,
    FIVR5KCSVRRetrieval,
    FIVR5KISVRRetrieval,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        np.save(stream, value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _instruction(
    model: EncoderProtocol, task: AbsTaskRetrieval, prompt_type: PromptType
) -> str | None:
    getter = getattr(model, "get_task_instruction", None)
    return getter(task.metadata, prompt_type) if getter else None


def _encode(
    model: EncoderProtocol,
    task: AbsTaskRetrieval,
    dataset: Dataset,
    *,
    prompt_type: PromptType,
    batch_size: int,
) -> np.ndarray:
    dataloader = create_dataloader(
        dataset,
        task_metadata=task.metadata,
        prompt_type=prompt_type,
        batch_size=batch_size,
    )
    return _as_numpy(
        model.encode(
            dataloader,
            task_metadata=task.metadata,
            hf_split="test",
            hf_subset="default",
            prompt_type=prompt_type,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    )


def _encode_checkpointed(
    model: EncoderProtocol,
    task: AbsTaskRetrieval,
    dataset: Dataset,
    *,
    prompt_type: PromptType,
    batch_size: int,
    chunk_size: int,
    shard_dir: Path,
) -> np.ndarray:
    shards: list[np.ndarray] = []
    for start in range(0, len(dataset), chunk_size):
        stop = min(start + chunk_size, len(dataset))
        shard_path = shard_dir / f"{start:06d}-{stop:06d}.npy"
        if shard_path.is_file():
            shard = np.load(shard_path)
            if len(shard) != stop - start:
                raise ValueError(f"invalid cached embedding shard: {shard_path}")
        else:
            shard = _encode(
                model,
                task,
                dataset.select(range(start, stop)),
                prompt_type=prompt_type,
                batch_size=batch_size,
            )
            _write_npy(shard_path, shard)
        shards.append(shard)
        print(f"encoded {stop}/{len(dataset)}", flush=True)
    return np.concatenate(shards)


def _score(
    task: AbsTaskRetrieval,
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    model: EncoderProtocol,
) -> dict[str, Any]:
    split = task.dataset["default"]["test"]
    similarities = _as_numpy(model.similarity(query_embeddings, corpus_embeddings))
    query_ids = split["queries"]["id"]
    corpus_ids = split["corpus"]["id"]
    results = {
        query_id: dict(zip(corpus_ids, row.astype(float), strict=True))
        for query_id, row in zip(query_ids, similarities, strict=True)
    }
    metrics = calculate_retrieval_scores(results, split["relevant_docs"], task.k_values)
    return make_score_dict(
        ndcg=metrics.ndcg,
        _map=metrics.map,
        recall=metrics.recall,
        precision=metrics.precision,
        mrr=metrics.mrr,
        naucs=metrics.naucs,
        naucs_mrr=metrics.naucs_mrr,
        hit_rate=metrics.hit_rate,
        task_scores={},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--embedding-dir", type=Path, default=Path(".fivr-embeddings"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.video_dir is None:
        configured_video_dir = os.environ.get("MTEB_FIVR_VIDEO_DIR")
        if configured_video_dir is None:
            raise ValueError("set --video-dir or MTEB_FIVR_VIDEO_DIR")
        args.video_dir = Path(configured_video_dir)

    tasks = [task_class() for task_class in TASK_CLASSES]
    for task in tasks:
        task.load_data(
            fivr_metadata_dir=args.metadata_dir,
            fivr_video_dir=args.video_dir,
        )
    first_split = tasks[0].dataset["default"]["test"]
    corpus_ids = list(first_split["corpus"]["id"])
    query_ids = list(first_split["queries"]["id"])
    for task in tasks[1:]:
        split = task.dataset["default"]["test"]
        if (
            list(split["corpus"]["id"]) != corpus_ids
            or list(split["queries"]["id"]) != query_ids
        ):
            raise ValueError("FIVR task views do not share identical media/order")

    model = cast(
        "EncoderProtocol",
        mteb.get_model(
            args.model,
            device=args.device,
            fps=None,
            num_frames=args.num_frames,
        ),
    )
    model_meta = model.mteb_model_meta
    if model_meta is None:
        raise ValueError("model has no MTEB metadata")
    document_instructions = [
        _instruction(model, task, PromptType.document) for task in tasks
    ]
    if len(set(document_instructions)) != 1:
        raise ValueError("shared corpus has task-dependent document instructions")

    cache_spec = {
        "model": model_meta.name,
        "revision": model_meta.revision,
        "num_frames": args.num_frames,
        "document_instruction": document_instructions[0],
        "corpus_ids": corpus_ids,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_spec, sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_dir = args.embedding_dir / cache_key
    _write_json(cache_dir / "spec.json", cache_spec)
    corpus_embeddings = _encode_checkpointed(
        model,
        tasks[0],
        first_split["corpus"],
        prompt_type=PromptType.document,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        shard_dir=cache_dir / "corpus",
    )

    scores: dict[str, Any] = {}
    query_instructions: dict[str, str | None] = {}
    for task in tasks:
        query_instruction = _instruction(model, task, PromptType.query)
        query_instructions[task.metadata.name] = query_instruction
        query_cache_spec = {
            "task": task.metadata.name,
            "instruction": query_instruction,
            "query_ids": query_ids,
        }
        query_key = hashlib.sha256(
            json.dumps(query_cache_spec, sort_keys=True).encode()
        ).hexdigest()[:16]
        query_path = cache_dir / f"queries-{query_key}.npy"
        if query_path.is_file():
            query_embeddings = np.load(query_path)
        else:
            query_embeddings = _encode(
                model,
                task,
                task.dataset["default"]["test"]["queries"],
                prompt_type=PromptType.query,
                batch_size=args.batch_size,
            )
            _write_npy(query_path, query_embeddings)
        scores[task.metadata.name] = _score(
            task, query_embeddings, corpus_embeddings, model
        )

    payload = {
        "model": model_meta.name,
        "revision": model_meta.revision,
        "experiment_kwargs": model_meta.experiment_kwargs,
        "device": args.device,
        "num_frames": args.num_frames,
        "batch_size": args.batch_size,
        "corpus_size": len(corpus_ids),
        "queries": len(query_ids),
        "document_instruction": document_instructions[0],
        "query_instructions": query_instructions,
        "scores": scores,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
