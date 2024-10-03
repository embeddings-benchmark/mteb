from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikipediaEasy10Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikipediaEasy10Clustering",
        description="TBW",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaEasy10Clustering",
            "revision": "0a0886b06acbfc735bca6a71b21ce1e5cb92a37b",
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
