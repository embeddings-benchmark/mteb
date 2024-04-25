from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class TwentyNewsgroupsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="TwentyNewsgroupsClustering",
        description="Clustering of the 20 Newsgroups dataset (subject only).",
        reference="https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        dataset={
            "path": "mteb/twentynewsgroups-clustering",
            "revision": "6125ec4e24fa026cec8a478383ee943acfbd5449",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples={"test": 59545},
        avg_character_length={"test": 32.0},
    )
