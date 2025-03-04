from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeTransOceanDLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTransOceanDL",
        description="The dataset is a collection of equivalent Python Deep Learning code snippets written in different machine learning framework. The task is to retrieve the equivalent code snippet in another framework, given a query code snippet from one framework.",
        reference="https://arxiv.org/abs/2310.04951",
        dataset={
            "path": "CoIR-Retrieval/codetrans-dl",
            "revision": "281562cb8a1265ab5c0824bfa6ddcd9b0a15618f",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2023-10-08", "2023-10-08"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
              title={CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation},
              author={Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
              year={2023},
              eprint={2310.04951},
              archivePrefix={arXiv},
              primaryClass={cs.AI},
              url={https://arxiv.org/abs/2310.04951},
        }""",
    )
