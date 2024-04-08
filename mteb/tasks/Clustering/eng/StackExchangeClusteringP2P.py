from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class StackExchangeClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="StackExchangeClusteringP2P",
        description="Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/stackexchange-clustering-p2p",
            "revision": "815ca46b2622cec33ccafc3735d572c266efdb44",
        },
        type="Clustering",
        category="p2p",
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
        n_samples={"test": 75000},
        avg_character_length={"test": 1090.7},
    )
