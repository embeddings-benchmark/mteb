from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LocBenchReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LocBenchRR",
        description="Software Issue Localization.",
        reference="https://arxiv.org/abs/2503.09089",
        dataset={
            "path": "mteb/LocBenchRR",
            "revision": "4a1fe7be94481b1e5a47d072782c89b146b4fecf",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="recall_at_10",
        date=("2025-03-12", "2025-03-12"),  # arxiv v1 submission date
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a github issue, identify the code that needs to be changed to fix the issue."
        },
        bibtex_citation=r"""
@misc{chen2025locagentgraphguidedllmagents,
  archiveprefix = {arXiv},
  author = {Zhaoling Chen and Xiangru Tang and Gangda Deng and Fang Wu and Jialong Wu and Zhiwei Jiang and Viktor Prasanna and Arman Cohan and Xingyao Wang},
  eprint = {2503.09089},
  primaryclass = {cs.SE},
  title = {LocAgent: Graph-Guided LLM Agents for Code Localization},
  url = {https://arxiv.org/abs/2503.09089},
  year = {2025},
}
""",
    )
