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
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="v_measure",
        date=("1999-01-01", "2024-01-01"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 2048},
        avg_character_length={"test": 890.1},
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
