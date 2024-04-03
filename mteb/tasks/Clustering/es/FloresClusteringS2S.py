from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class FloresClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FloresClusteringS2S",
        description="Clustering of sentences from various web articles, 32 topics in total.",
        reference="https://huggingface.co/datasets/facebook/flores",
        dataset={
            "path": "facebook/flores",
            "revision": "480b580487f53a46f881354a8348335d4edbb2de",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["es"],
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
        n_samples=None,
        avg_character_length=None,
    )
