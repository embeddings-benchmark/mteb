from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeFeedbackST(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeFeedbackST",
        description="The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.",
        reference="https://arxiv.org/abs/2407.02883",
        dataset={
            "path": "CoIR-Retrieval/codefeedback-st",
            "revision": "d213819e87aab9010628da8b73ab4eb337c89340",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{li2024coircomprehensivebenchmarkcode,
  archiveprefix = {arXiv},
  author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
  eprint = {2407.02883},
  primaryclass = {cs.IR},
  title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
  url = {https://arxiv.org/abs/2407.02883},
  year = {2024},
}
""",
    )
