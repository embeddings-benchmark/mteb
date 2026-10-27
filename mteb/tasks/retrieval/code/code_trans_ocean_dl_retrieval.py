from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeTransOceanDLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTransOceanDL",
        description="The dataset is a collection of equivalent Python Deep Learning code snippets written in different machine learning framework. The task is to retrieve the equivalent code snippet in another framework, given a query code snippet from one framework.",
        reference="https://aclanthology.org/2023.findings-emnlp.337/",
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
@inproceedings{yan2023codetransoceancomprehensivemultilingualbenchmark,
  address = {Singapore},
  author = {Yan, Weixiang and Tian, Yuchen and Li, Yunzhe and Chen, Qian and Wang, Wen},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2023},
  doi = {10.18653/v1/2023.findings-emnlp.337},
  editor = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  month = {dec},
  pages = {5067--5089},
  publisher = {Association for Computational Linguistics},
  title = {{C}ode{T}rans{O}cean: A Comprehensive Multilingual Benchmark for Code Translation},
  url = {https://aclanthology.org/2023.findings-emnlp.337/},
  year = {2023},
}
""",
    )
