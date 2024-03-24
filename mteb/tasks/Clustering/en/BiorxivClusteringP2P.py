from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class BiorxivClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BiorxivClusteringP2P",
        description="Clustering of titles+abstract from biorxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.biorxiv.org/",
        hf_hub_name="mteb/biorxiv-clustering-p2p",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="65b79d1d13f80053f67aca9498d9402c2d9f1f40",
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
        avg_character_length={"test": 1666.2},
    )
