from __future__ import annotations

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast, convert_to_fast
from mteb.abstasks.TaskMetadata import TaskMetadata


class TenKGnadClusteringP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="TenKGnadClusteringP2P.v2",
        description="Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-p2p",
            "revision": "5c59e41555244b7e45c9a6be2d720ab4bafae558",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=(
            "2000-01-01",
            "2020-12-31",
        ),  # since it is news it is guessed that it is from 2000 to 2020
        form=["written"],
        domains=["News", "Non-fiction"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,  # none found
        n_samples={"test": 10275},  # due to duplicates
        avg_character_length={"test": 2641.03},
    )

    def dataset_transform(self) -> None:
        ds = convert_to_fast(self.dataset, self.seed)  # type: ignore
        self.dataset = ds
