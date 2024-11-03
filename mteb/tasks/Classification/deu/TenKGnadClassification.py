from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TenKGnadClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TenKGnadClassification",
        description="10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "community-datasets/gnad10",
            "revision": "0798affe9b3f88cfda4267b6fbc50fac67046ee5",
            "trust_remote_code": True,
        },
        type="Classification",
        category="p2p",
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
        bibtex_citation="""
            @InProceedings{Schabus2017,
                Author    = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
                Title     = {One Million Posts: A Data Set of German Online Discussions},
                Booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
                Pages     = {1241--1244},
                Year      = {2017},
                Address   = {Tokyo, Japan},
                Doi       = {10.1145/3077136.3080711},
                Month     = aug
                }
        """,
    )
