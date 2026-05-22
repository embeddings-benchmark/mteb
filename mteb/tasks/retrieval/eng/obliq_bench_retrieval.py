from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_SUBSETS = [
    "twitter",
    "wildchat",
    "math",
    "writing",
    "congress",
]

# Subsets that ship per-query corpus exclusions in the original release.
_EXCLUSION_SUBSETS = ("math", "writing")


class OBLIQBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OBLIQBenchRetrieval",
        description=(
            "OBLIQ-Bench is a suite of five retrieval benchmarks for oblique queries, "
            "where the attributes that determine relevance are latent and have little "
            "surface expression in documents. The five subsets span implicit stance in "
            "tweets (twitter), AI failure modes in chat logs (wildchat), matched proof "
            "strategies across math sub-fields (math), writing-style author attribution "
            "(writing), and tip-of-tongue recollection of congressional hearings "
            "(congress). The benchmark isolates an asymmetry where reasoning LLMs can "
            "verify latent relevance once a document is surfaced, but retrievers fail "
            "to surface it in the first place. For the math and writing subsets, the "
            "original release ships per-query exclusion lists (same-source documents "
            "that the paper drops from the candidate pool before scoring); this task "
            "applies them via per-query top_ranked candidate restrictions."
        ),
        reference="https://arxiv.org/abs/2605.06235",
        dataset={
            "path": "mteb/OBLIQBenchRetrieval",
            "revision": "de1f23fbf7e9f8a4510d6bb17db5eb417c1eb3c6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={subset: ["eng-Latn"] for subset in _SUBSETS},
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-05-09"),
        domains=["Social", "Web", "Academic", "Government", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@article{tchuindjo2026obliq,
  author = {Tchuindjo, Diane and Shah, Devavrat and Khattab, Omar},
  journal = {arXiv preprint arXiv:2605.06235},
  title = {OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers with Latent and Implicit Queries},
  year = {2026},
}""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return
        super().load_data(num_proc=num_proc, **kwargs)
        for subset in _EXCLUSION_SUBSETS:
            if subset not in self.dataset:
                continue
            split_data = self.dataset[subset]["test"]
            corpus_ids = list(split_data["corpus"]["id"])
            excluded_ds = load_dataset(
                self.metadata.dataset["path"],
                name=f"{subset}-excluded",
                split="test",
                revision=self.metadata.dataset["revision"],
            )
            excluded = dict(
                zip(excluded_ds["query-id"], excluded_ds["excluded-corpus-ids"])
            )
            split_data["top_ranked"] = {
                qid: [cid for cid in corpus_ids if cid not in set(excl)]
                for qid, excl in excluded.items()
            }
