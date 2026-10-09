"""Evaluate shared ViDoRe candidate sources and export the proposed result format."""

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
    configurations = set()
    for domain in args.domains:
        for first_stage in args.first_stages:
            task = mteb.get_task(
                f"Vidore3{domain}Retrieval.v2", hf_subsets=args.subsets
            )
            assert isinstance(task, AbsTaskRetrieval)
            task.convert_to_reranking(first_stage=first_stage, top_k=args.top_k)
            result = mteb.evaluate(model, task, cache=cache, co2_tracker=False)
            configuration = result[0].reranking
            assert configuration is not None
            configurations.add(configuration.configuration_id)
            tasks.append(task.metadata.name)
            print(domain, first_stage, result[0].get_score())

    # Reload the files that a results-repository submission would contain.
    results = cache.load_results(
        models=[model.mteb_model_meta],
        tasks=[
            mteb.get_task(name, hf_subsets=args.subsets) for name in sorted(set(tasks))
        ],
        include_remote=False,
        only_main_score=True,
    )
    rows = results._to_dataset()
    # The output directory may also contain earlier runs with other depths/pins.
    rows = rows.filter(lambda row: row["reranking_id"] in configurations)
    rows.to_parquet(args.output / "reranking-results.parquet")
    rows.to_json(args.output / "reranking-results.jsonl")
    for configuration_id in sorted(configurations):
        print(configuration_id)
        print(results.filter_reranking(configuration_id).to_dataframe())
    print(f"Saved result JSONs and exports in {args.output.resolve()}")


if __name__ == "__main__":
    main()
