"""Run OBLIQBenchRetrieval with a single model. CLI: python run_eval.py <model> [out]."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mteb


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_eval.py <model_name> [output_folder]")
    model_name = sys.argv[1]
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results_obliq")
    output_folder.mkdir(parents=True, exist_ok=True)

    task = mteb.get_task("OBLIQBenchRetrieval")
    model = mteb.get_model(model_name)

    # Cap sequence length on MPS to avoid 20+ GiB attention buffer OOM on long
    # WildChat conversations (avg ~4k chars, Qwen3 default is 32k tokens).
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "max_seq_length"):
        inner.max_seq_length = 2048

    eval = mteb.MTEB(tasks=[task])
    results = eval.run(
        model,
        output_folder=str(output_folder),
        overwrite_results=True,
        encode_kwargs={"batch_size": 4},
    )

    summary: dict[str, float] = {}
    for r in results:
        per_subset = r.scores.get("test", {})
        for sub_entry in per_subset:
            sub = sub_entry.get("hf_subset", "main")
            summary[sub] = sub_entry.get("ndcg_at_10")
    print(json.dumps({model_name: summary}, indent=2))


if __name__ == "__main__":
    main()
