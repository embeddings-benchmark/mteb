from mteb.abstasks.clustering import AbsTaskClustering, _convert_to_fast
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class TenKGnadClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="TenKGnadClusteringP2P",
        description="Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-p2p",
            "revision": "5c59e41555244b7e45c9a6be2d720ab4bafae558",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=None,
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation="found",
        bibtex_citation=None,
        superseded_by="TenKGnadClusteringP2P.v2",
    )


class TenKGnadClusteringP2PFast(AbsTaskClustering):
    max_document_to_embed = 10275
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="TenKGnadClusteringP2P.v2",
        description="Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-p2p",
            "revision": "5c59e41555244b7e45c9a6be2d720ab4bafae558",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=(
            "2000-01-01",
            "2020-12-31",
        ),  # since it is news it is guessed that it is from 2000 to 2020
        domains=["News", "Non-fiction", "Written"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=None,  # none found
        # due to duplicates
        adapted_from=["TenKGnadClusteringP2P"],
    )

    def dataset_transform(self) -> None:
        ds = _convert_to_fast(
            self.dataset, self.input_column_name, self.label_column_name, self.seed
        )
        self.dataset = ds
