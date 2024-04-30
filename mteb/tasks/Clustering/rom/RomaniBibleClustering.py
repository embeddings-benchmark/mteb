from __future__ import annotations

from mteb.abstasks import AbsTaskClustering, TaskMetadata


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
        eval_splits=["test"],
        eval_langs=["rom-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2020-12-31"),
        form=["written"],
        domains=["Religious"],
        task_subtypes=["Thematic clustering"],
        license="MIT",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=["Kalderash"],
        text_creation="human-translated and localized",
        bibtex_citation=None,
        n_samples={"test": 2048},
        avg_character_length={"test": 132.2},
    )
