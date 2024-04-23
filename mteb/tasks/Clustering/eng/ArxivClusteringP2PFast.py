from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArxivClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ArxivClusteringP2P.v2",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "a122ad7f3f0291bf49cc6f4d32aa80929df69d5d",
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
        n_samples={"test": 250_000},
        avg_character_length={"test": 1009.98},
    )

    def dataset_transform(self):
        sent = self.dataset["test"]["sentences"][0]  # type: ignore
        lab = self.dataset["test"]["labels"][0]  # type: ignore

        self.dataset["test"] = datasets.Dataset.from_dict(  # type: ignore
            {"sentences": sent, "labels": lab}
        )
