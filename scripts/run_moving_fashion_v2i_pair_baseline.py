#!/usr/bin/env python3
"""Run a multimodal baseline on MovingFashion V2I pair classification.

The default uses EBind's audio/vision checkpoint, which embeds images and
videos in the same space and samples eight video frames. Install its pinned
source dependency before running::

    pip install -r scripts/requirements_moving_fashion_v2i_ebind.txt

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
from mteb.models.model_implementations.ebind_models import (
    EBindWrapper,
    ebind_audio_vision,
)
from mteb.tasks.pair_classification.zxx.moving_fashion_pc import (
    MovingFashionV2IPairClassification,
)

_DEFAULT_MODEL = "encord-team/ebind-audio-vision"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"MTEB model name (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
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


def _load_model(
    model_name: str, device: str, model_kwargs: dict[str, int | float | None]
):
    if model_name != _DEFAULT_MODEL:
        return mteb.get_model(model_name, device=device, **model_kwargs)

    # MTEB cannot publish EBind's Git dependency as a package extra on PyPI.
    # Load the existing registered wrapper directly after the experiment-specific
    # requirements file has installed the dependency.
    try:
        model = EBindWrapper(
            model_name=model_name,
            revision=ebind_audio_vision.revision,
            device=device,
            **model_kwargs,
        )
    except ModuleNotFoundError as error:
        if error.name != "ebind":
            raise
        raise RuntimeError(
            "EBind is not installed. Run `pip install -r "
            "scripts/requirements_moving_fashion_v2i_ebind.txt` first."
        ) from error

    loader_kwargs = {
        **ebind_audio_vision.loader_kwargs,
        **model_kwargs,
        "device": device,
    }
    model.mteb_model_meta = ebind_audio_vision.model_copy(
        deep=True,
        update={
            "experiment_kwargs": model_kwargs or None,
            "loader_kwargs": loader_kwargs,
        },
    )
    return model


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

    print(f"Model: {args.model}")
    model = _load_model(args.model, args.device, model_kwargs)
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
