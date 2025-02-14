from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class GLDv2I2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="GLDv2I2IRetrieval",
        description="Retrieve names of landmarks based on their image.",
        reference="https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html",
        dataset={
            "path": "gowitheflow/gld-v2",
            "revision": "c6b162ee349adb293901128a18c0b446f7b43457",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@InProceedings{Weyand_2020_CVPR,
author = {Weyand, Tobias and Araujo, Andre and Cao, Bingyi and Sim, Jack},
title = {Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

""",
        descriptive_stats={
            "n_samples": {"test": 1129},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 847728,
                    "num_queries": 1129,
                    "average_relevant_docs_per_query": 13.49,
                }
            },
        },
    )
