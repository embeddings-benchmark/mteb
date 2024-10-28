from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


class RuSciBenchGRNTIClusteringP2P(AbsTaskClusteringFast):
    max_document_to_embed = 2048
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="RuSciBenchGRNTIClusteringP2P",
        dataset={
            # here we use the same split for clustering
            "path": "ai-forever/ru-scibench-grnti-classification",
            "revision": "673a610d6d3dd91a547a0d57ae1b56f37ebbf6a1",
        },
        description="Clustering of scientific papers (title+abstract) by rubric",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench/",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="v_measure",
        date=("1999-01-01", "2024-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 2048},
            "test": {
                "num_samples": 2048,
                "average_text_length": 889.81396484375,
                "average_labels_per_text": 1.0,
                "unique_labels": 28,
                "labels": {
                    "3": {"count": 73},
                    "4": {"count": 73},
                    "20": {"count": 73},
                    "9": {"count": 73},
                    "21": {"count": 73},
                    "15": {"count": 73},
                    "16": {"count": 74},
                    "2": {"count": 73},
                    "8": {"count": 73},
                    "23": {"count": 73},
                    "6": {"count": 73},
                    "24": {"count": 73},
                    "10": {"count": 73},
                    "1": {"count": 73},
                    "17": {"count": 74},
                    "14": {"count": 74},
                    "18": {"count": 73},
                    "27": {"count": 73},
                    "19": {"count": 73},
                    "22": {"count": 73},
                    "12": {"count": 73},
                    "25": {"count": 73},
                    "5": {"count": 74},
                    "0": {"count": 73},
                    "26": {"count": 73},
                    "11": {"count": 73},
                    "13": {"count": 73},
                    "7": {"count": 73},
                },
            },
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"label": "labels", "text": "sentences"}
        )

        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=["test"],
            n_samples=2048,
            label="labels",
        )
