"""Run the MetaWorld MT50 image-to-video and video-to-image retrieval tasks.

For a quick reproducibility check:

    python scripts/run_metaworld_mt50_retrieval.py

To evaluate a multimodal model instead:

    python scripts/run_metaworld_mt50_retrieval.py \\
        --model jinaai/jina-embeddings-v5-omni-nano --batch-size 1
"""

from __future__ import annotations

import argparse

from mteb import MTEB, get_model
from mteb.tasks.retrieval.zxx.metaworld_retrieval import (
    MetaWorldMT50I2VRetrieval,
    MetaWorldMT50V2IRetrieval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mteb/baseline-random-encoder",
        help="MTEB model name to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Encoding batch size. Use 1 for large multimodal models.",
    )
    parser.add_argument(
        "--device",
        help="Device passed to the MTEB model loader, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--output-folder",
        default="results",
        help="Directory in which MTEB stores result files.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Overwrite any existing results for this model and task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = get_model(args.model, device=args.device)
    evaluation = MTEB(
        tasks=[MetaWorldMT50I2VRetrieval(), MetaWorldMT50V2IRetrieval()]
    )
    evaluation.run(
        model,
        output_folder=args.output_folder,
        overwrite_results=args.overwrite_results,
        encode_kwargs={"batch_size": args.batch_size},
    )


if __name__ == "__main__":
    main()
