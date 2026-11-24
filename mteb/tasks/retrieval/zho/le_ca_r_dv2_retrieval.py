from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LeCaRDv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LeCaRDv2",
        description="The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.",
        reference="https://github.com/THUIR/LeCaRDv2",
        dataset={
            "path": "mteb/LeCaRDv2",
            "revision": "b78e18688c3d012a33dc3676597c1d1b2243ce1c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["zho-Hans"],
        main_score="ndcg_at_10",
        date=("2023-10-01", "2023-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{li2023lecardv2,
  author = {Li, Haitao and Shao, Yunqiu and Wu, Yueyue and Ai, Qingyao and Ma, Yixiao and Liu, Yiqun},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  collection = {SIGIR 2024},
  doi = {10.1145/3626772.3657887},
  month = {July},
  pages = {2251–2260},
  publisher = {ACM},
  series = {SIGIR 2024},
  title = {LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset},
  url = {https://doi.org/10.1145/3626772.3657887},
  year = {2024},
}
""",
    )
