from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikipediaMedium5Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikipediaMedium5Clustering",
        description="TBW",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Clustering_Medium_5_Class",
            "revision": "178f49f21672a31f3fc94ac28e5703eb7c8d3291",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=None,
        domains=None,
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )
