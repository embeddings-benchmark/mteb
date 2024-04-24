from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class BigPatentClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BigPatentClustering",
        description="Clustering of documents from the Big Patent dataset. Test set only includes documents"
        "belonging to a single category, with a total of 9 categories.",
        reference="https://www.kaggle.com/datasets/big_patent",
        dataset={
            "path": "jinaai/big-patent-clustering",
            "revision": "62d5330920bca426ce9d3c76ea914f15fc83e891",
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
        bibtex_citation="",
        n_samples=None,
        avg_character_length=None,
    )
