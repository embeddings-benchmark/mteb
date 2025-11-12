from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TenKGnadClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TenKGnadClassification",
        description="10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "mteb/TenKGnadClassification",
            "revision": "ae9862bbcddc27b4bd93e2a7b463b7b5d05c6c55",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="accuracy",
        date=("2015-06-01", "2016-05-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Schabus2017,
  address = {Tokyo, Japan},
  author = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
  booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  doi = {10.1145/3077136.3080711},
  month = aug,
  pages = {1241--1244},
  title = {One Million Posts: A Data Set of German Online Discussions},
  year = {2017},
}
""",
        superseded_by="TenKGnadClassification.v2",
    )


class TenKGnadClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TenKGnadClassification.v2",
        description="10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes). This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "mteb/ten_k_gnad",
            "revision": "fc6825fe0d813e7fc92f05fe63ac4bb3ee191c4d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="accuracy",
        date=("2015-06-01", "2016-05-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Schabus2017,
  address = {Tokyo, Japan},
  author = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
  booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  doi = {10.1145/3077136.3080711},
  month = aug,
  pages = {1241--1244},
  title = {One Million Posts: A Data Set of German Online Discussions},
  year = {2017},
}
""",
        adapted_from=["TenKGnadClassification"],
    )
