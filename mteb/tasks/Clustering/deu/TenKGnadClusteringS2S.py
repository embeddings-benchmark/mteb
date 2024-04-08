from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class TenKGnadClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="TenKGnadClusteringS2S",
        description="Clustering of news article titles. Clustering of 10 splits on the news article category.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-s2s",
            "revision": "6cddbe003f12b9b140aec477b583ac4191f01786",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["de"],
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
        n_samples={"test": 45914},
        avg_character_length={"test": 50.96},
    )
