from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikiClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories as defined in https://da.wikipedia.org/wiki/Kategori:Topniveau_for_emner.",
        reference=None,
        dataset={
            "path": "ryzzlestrizzle/dawiki-clustering-p2p",
            "revision": "4c4c6b6ad63311d6a072734d86dca1c0b48c441c",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 150_000},
        avg_character_length={"test": 750.34},
    )
