from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoStrategyQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Ko-StrategyQA",
        description="Ko-StrategyQA",
        reference=None,
        dataset={
            "path": "taeminlee/Ko-StrategyQA",
            "revision": "d243889a3eb6654029dbd7e7f9319ae31d58f97c",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@article{geva2021strategyqa,
  title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  year = {2021},
}""",
    )
