from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class RomaniBibleClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RomaniBibleClustering",
        description="Clustering verses from the Bible in Kalderash Romani by book.",
        reference="https://romani.global.bible/info",
        dataset={
            "path": "kardosdrur/romani-bible",
            "revision": "97fae0e80a8d275bc685dcb3da08972af542ad6e",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rom-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2020-12-31"),
        domains=["Religious", "Written"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        annotations_creators="derived",
        dialect=["Kalderash"],
        sample_creation="human-translated and localized",
        bibtex_citation=None,
    )
