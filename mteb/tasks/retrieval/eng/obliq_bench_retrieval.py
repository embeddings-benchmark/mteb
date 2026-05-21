from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_SUBSETS = [
    "twitter",
    "wildchat",
    "math",
    "writing",
    "congress",
]


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
            "to surface it in the first place. Note: the original release ships "
            "per_query_excluded_ids.json for the math and writing subsets (same-source "
            "snippets to drop per query before scoring); this MTEB integration does not "
            "apply those exclusions, so scores may differ slightly from the paper."
        ),
        reference="https://arxiv.org/abs/2605.06235",
        dataset={
            "path": "mteb/OBLIQBenchRetrieval",
            "revision": "5345ce643839c6826c267180738b16075add308d",
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
        prompt={
            "query": "Given an oblique query expressing a latent or implicit property, retrieve documents that instantiate that property."
        },
    )
