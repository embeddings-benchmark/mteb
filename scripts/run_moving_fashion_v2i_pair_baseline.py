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
is useful only for a faster diagnostic run. Full runs save pair-level scores and
automatically report random-ranking, source-subset, and video-cluster-bootstrap
analyses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

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

if __package__:
    from scripts.analyze_moving_fashion_v2i_pair_predictions import (
        analyze_predictions,
        print_summary,
        write_analysis,
    )
else:  # Support `python scripts/run_...py` from the repo root.
    from analyze_moving_fashion_v2i_pair_predictions import (  # type: ignore[import-not-found, no-redef]
        analyze_predictions,
        print_summary,
        write_analysis,
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
        "--prediction-folder",
        type=Path,
        help="Override the model-specific folder used for pair-level predictions.",
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
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=1_000)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "Skip model loading and encoding, then analyze existing pair-level "
            "predictions under --output-folder."
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.smoke_videos < 0:
        parser.error("--smoke-videos cannot be negative")
    if args.num_frames is not None and args.num_frames < 1:
        parser.error("--num-frames must be positive")
    if args.random_trials < 1:
        parser.error("--random-trials must be positive")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.analysis_only and args.smoke_videos:
        parser.error("--analysis-only cannot be combined with --smoke-videos")
    return args


def _load_model(model_name: str, device: str, model_kwargs: dict[str, Any]) -> Any:
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
    if task.dataset is None:
        raise RuntimeError("MovingFashion task did not load a dataset")
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


def _default_prediction_folder(output_folder: Path, model: Any) -> Path:
    metadata = model.mteb_model_meta
    model_name = str(metadata.name).replace("/", "__")
    revision = str(metadata.revision or "no_revision_available")
    experiment_kwargs = metadata.experiment_kwargs or {}
    experiment = (
        "__".join(f"{key}_{value}" for key, value in sorted(experiment_kwargs.items()))
        or "official_defaults"
    )
    return output_folder / "predictions" / model_name / revision / experiment


def _write_pair_manifest(
    task: MovingFashionV2IPairClassification, output_path: Path
) -> None:
    task.load_data()
    if task.dataset is None:
        raise RuntimeError("MovingFashion task did not load a dataset")
    dataset = task.dataset["test"]
    payload = {
        "task_name": task.metadata.name,
        "dataset_revision": task.metadata.dataset["revision"],
        "split": "test",
        "rows": {
            column: list(dataset[column])
            for column in ("video_id", "image_id", "label", "source_subset")
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _find_prediction_folder(
    output_folder: Path,
    prediction_folder: Path | None,
    prediction_file_name: str,
) -> Path:
    if prediction_folder is not None:
        return prediction_folder
    matches = list((output_folder / "predictions").rglob(prediction_file_name))
    if not matches:
        raise FileNotFoundError(
            f"No {prediction_file_name} found under {output_folder / 'predictions'}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple prediction files were found; select one with "
            f"--prediction-folder: {[str(path.parent) for path in matches]}"
        )
    return matches[0].parent


def _run_analysis(
    task: MovingFashionV2IPairClassification,
    prediction_folder: Path,
    *,
    random_trials: int,
    bootstrap_samples: int,
    seed: int,
) -> None:
    predictions_path = prediction_folder / task.prediction_file_name
    if not predictions_path.exists():
        raise FileNotFoundError(f"Pair-level predictions not found: {predictions_path}")
    pairs_path = prediction_folder / f"{task.metadata.name}_pairs.json"
    analysis_path = prediction_folder / f"{task.metadata.name}_analysis.json"
    _write_pair_manifest(task, pairs_path)
    analysis = analyze_predictions(
        predictions_path,
        pairs_path,
        random_trials=random_trials,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_analysis(analysis, analysis_path)
    print_summary(analysis)
    print(f"predictions_file={predictions_path.resolve()}")
    print(f"pairs_file={pairs_path.resolve()}")
    print(f"analysis_file={analysis_path.resolve()}")


def main() -> None:
    args = _parse_args()
    task = MovingFashionV2IPairClassification()
    if args.analysis_only:
        analysis_prediction_folder = _find_prediction_folder(
            args.output_folder,
            args.prediction_folder,
            task.prediction_file_name,
        )
        _run_analysis(
            task,
            analysis_prediction_folder,
            random_trials=args.random_trials,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.analysis_seed,
        )
        return

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. In Colab, select Runtime > Change runtime "
                "type > GPU, reconnect, and rerun the script."
            )
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_kwargs: dict[str, Any] = {}
    if args.num_frames is not None:
        model_kwargs = {"fps": None, "num_frames": args.num_frames}

    print(f"Model: {args.model}")
    model = _load_model(args.model, args.device, model_kwargs)
    if args.smoke_videos:
        _select_smoke_rows(task, args.smoke_videos)

    cache = None if args.smoke_videos else ResultCache(args.output_folder)
    prediction_folder: Path | None = None
    if not args.smoke_videos:
        prediction_folder = args.prediction_folder or _default_prediction_folder(
            args.output_folder, model
        )
    result = mteb.evaluate(
        model,
        task,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        encode_kwargs={"batch_size": args.batch_size},
        co2_tracker=False,
        prediction_folder=prediction_folder,
    )
    task_result = result.task_results[0]
    prefix = "SMOKE ONLY - " if args.smoke_videos else ""
    print(f"{prefix}main_score={task_result.main_score:.6f}")
    print(f"evaluation_time_seconds={task_result.evaluation_time:.2f}")
    if cache is not None:
        print(f"results_folder={args.output_folder.resolve()}")
    if prediction_folder is not None:
        _run_analysis(
            task,
            prediction_folder,
            random_trials=args.random_trials,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.analysis_seed,
        )


if __name__ == "__main__":
    main()
