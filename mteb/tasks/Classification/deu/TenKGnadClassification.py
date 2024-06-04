from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class TenKGnadClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TenKGnadClassification",
        description="10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "community-datasets/gnad10",
            "revision": "0798affe9b3f88cfda4267b6fbc50fac67046ee5",
        },
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="accuracy",
        date=("2015-06-01", "2016-05-31"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
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
        n_samples={"test": 1028},
        avg_character_length={"test": 2627.31},
    )
