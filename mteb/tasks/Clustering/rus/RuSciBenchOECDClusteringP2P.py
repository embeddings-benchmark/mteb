from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RuSciBenchOECDClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RuSciBenchOECDClusteringP2P",
        dataset={
            "path": "ai-forever/ru-scibench-oecd-clustering-p2p",
            "revision": "08475cf0f71cd474bdc3525ee49d8495a12a9a6a",
        },
        description="Clustering of scientific papers (title+abstract) by rubric",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench/",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="v_measure",
        date=("1999-01-01", "2024-01-01"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""""",
        n_samples={"test": 30740},
        avg_character_length={"test": 838.7},
    )
