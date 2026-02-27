from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class RomaniBibleClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="RomaniBibleClustering",
        description="Clustering verses from the Bible in Kalderash Romani by book.",
        reference="https://romani.global.bible/info",
        dataset={
            "path": "mteb/RomaniBibleClustering",
            "revision": "53db5afec1fe573b334cb6f1c8ee64a0849b3ce5",
        },
        type="Clustering",
        category="t2c",
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
        bibtex_citation="",
    )
