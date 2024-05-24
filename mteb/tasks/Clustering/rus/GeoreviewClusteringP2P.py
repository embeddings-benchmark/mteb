from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class GeoreviewClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="GeoreviewClusteringP2P",
        dataset={
            "path": "ai-forever/georeview-clustering-p2p",
            "revision": "e82bdbb7d767270d37c9b4ea88cb6475facfd656",
        },
        description="Review clustering based on Yandex Georeview dataset",
        reference="https://github.com/yandex/geo-reviews-dataset-2023",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="v_measure",
        date=("2023-01-01", "2023-07-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=[],
        license="mit",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""""",
        n_samples={"test": 301510},
        avg_character_length={"test": 290.5},
    )
