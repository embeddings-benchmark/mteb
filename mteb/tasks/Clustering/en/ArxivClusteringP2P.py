from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ArxivClusteringP2P",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        hf_hub_name="mteb/arxiv-clustering-p2p",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
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
        n_samples={"test": 732723},
        avg_character_length={"test": 1009.98},
    )
