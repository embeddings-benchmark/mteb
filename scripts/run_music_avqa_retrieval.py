"""Run the MUSIC-AVQA audio/video retrieval tasks with a registered MTEB model.

Example:
    python scripts/run_music_avqa_retrieval.py --model jinaai/jina-embeddings-v5-omni
"""

from __future__ import annotations

import argparse

from mteb import MTEB, get_model
from mteb.tasks.retrieval.eng.music_avqa_retrieval import (
    MusicAVQAA2VRetrieval,
    MusicAVQAV2ARetrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MTEB model identifier")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-folder", default="results")
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    model_kwargs = {"device": args.device} if args.device else {}
    model = get_model(args.model, **model_kwargs)
    evaluation = MTEB(tasks=[MusicAVQAA2VRetrieval(), MusicAVQAV2ARetrieval()])
    evaluation.run(
        model,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        overwrite_results=args.overwrite_results,
    )


if __name__ == "__main__":
    main()
