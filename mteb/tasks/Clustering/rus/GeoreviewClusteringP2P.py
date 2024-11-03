from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class GeoreviewClusteringP2P(AbsTaskClusteringFast):
    max_document_to_embed = 2000
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="GeoreviewClusteringP2P",
        dataset={
            "path": "ai-forever/georeview-clustering-p2p",
            "revision": "97a313c8fc85b47f13f33e7e9a95c1ad888c7fec",
        },
        description="Review clustering based on Yandex Georeview dataset",
        reference="https://github.com/yandex/geo-reviews-dataset-2023",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="v_measure",
        date=("2023-01-01", "2023-07-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify the organization category based on the reviews",
    )
