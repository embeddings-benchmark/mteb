from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeTransOceanContestRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTransOceanContest",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet",
        reference="https://aclanthology.org/2023.findings-emnlp.337/",
        dataset={
            "path": "CoIR-Retrieval/codetrans-contest",
            "revision": "20da4eb20a4b17300c0986ee148c90867a7f2a4d",
        },
        type="Retrieval",
        category="t2t",
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
