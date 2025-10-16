from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
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
        bibtex_citation=r"""
@article{geva2021strategyqa,
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
  year = {2021},
}
""",
    )
