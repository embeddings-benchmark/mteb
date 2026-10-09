"""Evaluate prepared ViDoRe candidates using existing model experiments."""

import argparse
from pathlib import Path

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mteb/baseline-random-encoder",
        help="Reranker model ID; the default only checks the workflow on CPU.",
    )
    parser.add_argument("--domains", nargs="+", default=["Hr", "Energy"])
    parser.add_argument(
        "--first-stages",
        nargs="+",
        default=["bm25-text", "bge-text", "qwen-text-image"],
    )
    parser.add_argument("--subsets", nargs="+", default=["english"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("vidore-reranking-output"))
    args = parser.parse_args()
    cache = mteb.ResultCache(args.output)
    model = mteb.get_model(args.model)
    tasks = []
    experiments = {}
    for domain in args.domains:
        for first_stage in args.first_stages:
            task = mteb.get_task(
                f"Vidore3{domain}Retrieval.v2", hf_subsets=args.subsets
            )
            assert isinstance(task, AbsTaskRetrieval)
            task.convert_to_reranking(first_stage=first_stage, top_k=args.top_k)
            result = mteb.evaluate(model, task, cache=cache, co2_tracker=False)
            experiments[result.experiment_name] = result.model_meta
            tasks.append(task.metadata.name)
            print(domain, first_stage, result[0].get_score(), result.experiment_name)

    # Use the returned metadata to select exactly the experiments just evaluated.
    results = cache.load_results(
        models=list(experiments.values()),
        tasks=[
            mteb.get_task(name, hf_subsets=args.subsets) for name in sorted(set(tasks))
        ],
        include_remote=False,
        only_main_score=True,
    )
    rows = results._to_dataset()
    rows.to_parquet(args.output / "reranking-results.parquet")
    rows.to_json(args.output / "reranking-results.jsonl")
    print(f"Saved result JSONs and exports in {args.output.resolve()}")


if __name__ == "__main__":
    main()
