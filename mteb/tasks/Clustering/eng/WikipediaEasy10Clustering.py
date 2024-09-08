from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikipediaEasy10Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikipediaEasy10Clustering",
        description="TBW",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Clustering_Easy_10_Class",
            "revision": "67aa9e201b030038d16241a39795f5b5e5a89898",
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
