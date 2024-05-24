from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RuSciBenchGRNTIClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RuSciBenchGRNTIClusteringP2P",
        dataset={
            "path": "ai-forever/ru-scibench-grnti-clustering-p2p",
            "revision": "5add37c2d5028dda82cf115a659b56153580c203",
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
        n_samples={"test": 31080},
        avg_character_length={"test": 863.3},
    )
