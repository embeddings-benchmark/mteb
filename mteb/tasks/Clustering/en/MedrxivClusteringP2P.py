from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class MedrxivClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MedrxivClusteringP2P",
        description="Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.medrxiv.org/",
        hf_hub_name="mteb/medrxiv-clustering-p2p",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="e7a26af6f3ae46b30dde8737f02c07b1505bcc73",
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
    )
