"""Run the full ViDoRe(v3) benchmark for a ColVec1 model via MTEB.

Usage::
    python run_vdrv3.py --model webAI-Official/webAI-ColVec1-4b
    python run_vdrv3.py --model webAI-Official/webAI-ColVec1-9b --batch-size 2
"""

from __future__ import annotations

import argparse
import logging

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
    parser.add_argument("--output-dir", default="./mteb_results", help="Results directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Model:      %s", args.model)
    logger.info("Output dir: %s", args.output_dir)

    model = mteb.get_model(args.model)
    tasks = mteb.get_benchmark("ViDoRe(v3)")

    cache = mteb.ResultCache(cache_path=args.output_dir)

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=cache,
        encode_kwargs={"batch_size": args.batch_size},
    )

    for result in results:
        logger.info("%s: %s", result.task_name, result.scores)

    logger.info("Results saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
