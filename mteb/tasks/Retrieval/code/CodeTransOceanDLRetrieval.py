from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.retrieval import AbsTaskRetrieval

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
        category="t2t",
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
