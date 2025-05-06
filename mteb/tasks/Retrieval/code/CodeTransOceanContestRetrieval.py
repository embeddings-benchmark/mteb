from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeTransOceanContestRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTransOceanContest",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet",
        reference="https://arxiv.org/abs/2310.04951",
        dataset={
            "path": "CoIR-Retrieval/codetrans-contest",
            "revision": "20da4eb20a4b17300c0986ee148c90867a7f2a4d",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code", "c++-Code"],
        main_score="ndcg_at_10",
        date=("2023-10-08", "2023-10-08"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
  archiveprefix = {arXiv},
  author = {Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
  eprint = {2310.04951},
  primaryclass = {cs.AI},
  title = {CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation},
  url = {https://arxiv.org/abs/2310.04951},
  year = {2023},
}
""",
    )
