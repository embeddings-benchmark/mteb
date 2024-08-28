from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class BiorxivClusteringS2SFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="BiorxivClusteringS2S.v2",
        description="Clustering of titles from biorxiv across 26 categories.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-s2s",
            "revision": "eb4edb10386758d274cd161093eb351381a16dbf",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 2151},
            "avg_character_length": {"test": 101.7},
        },
    )

    def dataset_transform(self):
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])


class BiorxivClusteringS2S(AbsTaskClustering):
    superseded_by = "BiorxivClusteringS2S.v2"
    metadata = TaskMetadata(
        name="BiorxivClusteringS2S",
        description="Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/biorxiv-clustering-s2s",
            "revision": "258694dd0231531bc1fd9de6ceb52a0853c6d908",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 75000},
            "test": {
                "num_samples": 10,
                "average_text_length": 7500.0,
                "average_labels_per_text": 7500.0,
                "unique_labels": 26,
                "labels": {
                    "neuroscience": {"count": 14251},
                    "genetics": {"count": 2282},
                    "biophysics": {"count": 3864},
                    "animal behavior and cognition": {"count": 1148},
                    "genomics": {"count": 3422},
                    "systems biology": {"count": 1544},
                    "ecology": {"count": 3469},
                    "immunology": {"count": 3517},
                    "evolutionary biology": {"count": 3756},
                    "molecular biology": {"count": 2772},
                    "bioengineering": {"count": 2169},
                    "cancer biology": {"count": 2922},
                    "plant biology": {"count": 2640},
                    "microbiology": {"count": 7176},
                    "physiology": {"count": 1251},
                    "synthetic biology": {"count": 686},
                    "pharmacology and toxicology": {"count": 864},
                    "zoology": {"count": 433},
                    "bioinformatics": {"count": 6294},
                    "cell biology": {"count": 4433},
                    "developmental biology": {"count": 2352},
                    "biochemistry": {"count": 2790},
                    "scientific communication and education": {"count": 349},
                    "paleontology": {"count": 120},
                    "pathology": {"count": 495},
                    "epidemiology": {"count": 1},
                },
            },
        },
    )
