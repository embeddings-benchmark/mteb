#!/usr/bin/env python3
"""Run a multimodal baseline on MovingFashion V2I pair classification.

The default invocation is intended for a Colab GPU runtime::

    python scripts/run_moving_fashion_v2i_pair_baseline.py \
        --output-folder /content/mteb-results

Use ``--smoke-videos`` to validate the environment without caching a partial
result. Omit ``--num-frames`` for the official model configuration; setting it
is useful only for a faster diagnostic run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

import mteb
from mteb.cache import ResultCache
from mteb.tasks.pair_classification.zxx.moving_fashion_pc import (
    MovingFashionV2IPairClassification,
)

_DEFAULT_MODEL = "jinaai/jina-embeddings-v5-omni-nano"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("results/moving-fashion-v2i-pair"),
    )
    parser.add_argument(
        "--overwrite-strategy",
        choices=("always", "never", "only-missing"),
        default="only-missing",
    )
    parser.add_argument(
        "--smoke-videos",
        type=int,
        default=0,
        help="Evaluate only this many video IDs and do not cache the partial result.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help=(
            "Override video sampling with a fixed frame count. Leave unset for "
            "the model's official configuration."
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.smoke_videos < 0:
        parser.error("--smoke-videos cannot be negative")
    if args.num_frames is not None and args.num_frames < 1:
        parser.error("--num-frames must be positive")
    return args


def _select_smoke_rows(
    task: MovingFashionV2IPairClassification, number_of_videos: int
) -> None:
    task.load_data()
    dataset = task.dataset["test"]
    selected_video_ids = list(dict.fromkeys(dataset["video_id"]))[:number_of_videos]
    selected_video_ids_set = set(selected_video_ids)
    indices = [
        index
        for index, video_id in enumerate(dataset["video_id"])
        if video_id in selected_video_ids_set
    ]
    task.dataset["test"] = dataset.select(indices)
    print(
        "Smoke selection: "
        f"{len(selected_video_ids)} videos, {len(indices)} rows, "
        f"labels={dict(sorted(Counter(task.dataset['test']['label']).items()))}"
    )


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. In Colab, select Runtime > Change runtime "
                "type > GPU, reconnect, and rerun the script."
            )
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_kwargs = {}
    if args.num_frames is not None:
        model_kwargs = {"fps": None, "num_frames": args.num_frames}

    model = mteb.get_model(args.model, device=args.device, **model_kwargs)
    task = MovingFashionV2IPairClassification()
    if args.smoke_videos:
        _select_smoke_rows(task, args.smoke_videos)

    cache = None if args.smoke_videos else ResultCache(args.output_folder)
    result = mteb.evaluate(
        model,
        task,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        encode_kwargs={"batch_size": args.batch_size},
        co2_tracker=False,
    )
    task_result = result.task_results[0]
    prefix = "SMOKE ONLY - " if args.smoke_videos else ""
    print(f"{prefix}main_score={task_result.main_score:.6f}")
    print(f"evaluation_time_seconds={task_result.evaluation_time:.2f}")
    if cache is not None:
        print(f"results_folder={args.output_folder.resolve()}")


if __name__ == "__main__":
    main()
