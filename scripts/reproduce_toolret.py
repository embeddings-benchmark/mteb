"""Reproduce Table 5 of the ToolRet paper (arXiv:2503.01763) with MTEB.

Three things have to line up for the published numbers to come out, and all three
are easy to get wrong:

1. Every query retrieves from the *pooled* 44,453-tool corpus, not from its own
   category's tools. `ToolRetrieval*` does this by sharing one corpus across subsets.
2. A subset score is the unweighted mean over the 35 retrieval tasks, not a
   micro-average over queries. The tasks are the MTEB subsets, so MTEB's
   cross-subset mean already matches; this script only regroups them for reporting.
3. `all-MiniLM-L6-v2` ships `max_seq_length=256`, while the reference
   implementation uses `min(max_position_embeddings, 2048)` = 512. The `web`
   subset has the longest tool documents and swings ~5 NDCG@10 on this alone.

Run with `python scripts/reproduce_toolret.py`. Wrapping the model in
`CachedEmbeddingWrapper` is worth it: the 35 subsets share a corpus, so without a
cache it is encoded 35 times.
"""

from __future__ import annotations

import collections

import mteb
from mteb.models import CachedEmbeddingWrapper
from mteb.tasks.retrieval.eng.tool_retrieval import _TASK_2_CATEGORY

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_MAX_SEQ_LENGTH = 512

# NDCG@10 x100, `all-MiniLM-L6-v2` row of Table 5 (w/ inst.)
PAPER_TABLE_5 = {"web": 12.77, "code": 31.59, "customized": 32.24}


def main() -> None:
    model = mteb.get_model(MODEL_NAME)
    model.model.max_seq_length = REFERENCE_MAX_SEQ_LENGTH

    task = mteb.get_tasks(tasks=["ToolRetrievalInstruction"])[0]
    results = mteb.evaluate(
        CachedEmbeddingWrapper(model, cache_path="toolret_cache"),
        [task],
        cache=None,
    )

    per_task = {
        entry["hf_subset"]: entry["ndcg_at_10"] * 100
        for result in results
        for entries in result.scores.values()
        for entry in entries
    }

    by_category = collections.defaultdict(list)
    for task_name, score in per_task.items():
        by_category[_TASK_2_CATEGORY[task_name]].append(score)

    print(f"\n{MODEL_NAME}, w/ inst., NDCG@10 x100\n")
    print(f"{'subset':12s} {'ours':>7s} {'paper':>7s} {'delta':>7s}")
    deltas = []
    for category in ("web", "code", "customized"):
        scores = by_category[category]
        ours = sum(scores) / len(scores)
        paper = PAPER_TABLE_5[category]
        deltas.append(abs(ours - paper))
        print(f"{category:12s} {ours:7.2f} {paper:7.2f} {ours - paper:+7.2f}")
    print(f"\nmean |delta| = {sum(deltas) / len(deltas):.2f}")


if __name__ == "__main__":
    main()
