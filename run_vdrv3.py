"""Run the full ViDoRe(v3) benchmark for a ColVec1 model via MTEB.

Usage::

    python run_vdrv3.py --model webAI-Official/webAI-ColVec1-4b
    python run_vdrv3.py --model webAI-Official/webAI-ColVec1-9b --batch-size 2
    python run_vdrv3.py --model webAI-Official/webAI-ColVec1-4b \
        --cache-dir ~/.cache/mteb --output-dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mteb

logger = logging.getLogger(__name__)


def main():
    """Load a model and evaluate it on ViDoRe(v3)."""
    parser = argparse.ArgumentParser(description="Run ViDoRe(v3) benchmark")
    parser.add_argument(
        "--model",
        required=True,
        help="HF Hub repo ID or local path to the model",
    )
    parser.add_argument(
        "--cache-dir",
        default=Path.home() / ".cache" / "mteb",
        help="MTEB result cache directory (default: ~/.cache/mteb)",
    )
    parser.add_argument(
        "--output-dir",
        default="./mteb_results",
        help="Directory to write final result JSON files",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Model:      %s", args.model)
    logger.info("Cache dir:  %s", args.cache_dir or "(default ~/.cache/mteb)")
    logger.info("Output dir: %s", args.output_dir)

    model = mteb.get_model(args.model)
    tasks = mteb.get_benchmark("ViDoRe(v3)")
    cache = mteb.ResultCache(cache_path=args.cache_dir)

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=cache,
        encode_kwargs={"batch_size": args.batch_size},
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        out_path = output_dir / f"{result.task_name}.json"
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("  Saved: %s", out_path)

    for result in results:
        logger.info("%s: %s", result.task_name, result.scores)


if __name__ == "__main__":
    main()
