from mteb.abstasks.clustering import AbsTaskClustering, _convert_to_fast
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class TenKGnadClusteringS2S(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="TenKGnadClusteringS2S",
        description="Clustering of news article titles. Clustering of 10 splits on the news article category.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-s2s",
            "revision": "6cddbe003f12b9b140aec477b583ac4191f01786",
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
        task_subtypes=["Thematic clustering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # none found
        superseded_by="TenKGnadClusteringS2S.v2",
    )


class TenKGnadClusteringS2SFast(AbsTaskClustering):
    max_document_to_embed = 10267
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="TenKGnadClusteringS2S.v2",
        description="Clustering of news article titles. Clustering of 10 splits on the news article category. v2 uses a faster evaluation method used in the MMTEB paper, which allow for notably faster evaluation.",
        reference="https://tblock.github.io/10kGNAD/",
        dataset={
            "path": "slvnwhrl/tenkgnad-clustering-s2s",
            "revision": "6cddbe003f12b9b140aec477b583ac4191f01786",
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
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # none found
        adapted_from=["TenKGnadClusteringS2S"],
    )

    def dataset_transform(self, num_proc: int = 1, **kwargs) -> None:
        ds = _convert_to_fast(
            self.dataset, self.input_column_name, self.label_column_name, self.seed
        )
        self.dataset = ds
