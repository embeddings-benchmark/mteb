from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArxivClusteringP2PFast(AbsTaskClustering):
    # a faster version of the dataset, since it does not sample from the same distribution we can't use the AbsTaskClusteringFast, instead we
    # simply downsample each cluster.

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
        ds = clustering_downsample(self.dataset, self.seed)
        self.dataset = ds
