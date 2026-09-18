from typing import Any

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

N_SAMPLES = 2048


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(",")[:2]
    return record


class VGHierarchicalClusteringP2P(AbsTaskClustering):
    max_document_to_embed = N_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="VGHierarchicalClusteringP2P",
        superseded_by="VGHierarchicalClusteringP2P.v2",
        dataset={
            "path": "navjordj/VG_summarization",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
        },
        description="Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.",
        reference="https://huggingface.co/datasets/navjordj/VG_summarization",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
        prompt="Identify the categories (e.g. sports) of given articles in Norwegian",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = self.dataset.rename_columns(
            {"article": "sentences", "classes": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
        # Subsampling the dataset
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=N_SAMPLES, seed=self.seed
        )["test"]


class VGHierarchicalClusteringS2S(AbsTaskClustering):
    max_document_to_embed = N_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="VGHierarchicalClusteringS2S",
        superseded_by="VGHierarchicalClusteringS2S.v2",
        dataset={
            "path": "navjordj/VG_summarization",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
        },
        description="Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.",
        reference="https://huggingface.co/datasets/navjordj/VG_summarization",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
        prompt="Identify the categories (e.g. sports) of given articles in Norwegian",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = self.dataset.rename_columns(
            {"ingress": "sentences", "classes": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
        # Subsampling the dataset
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=N_SAMPLES, seed=self.seed
        )["test"]


class VGHierarchicalClusteringP2PV2(VGHierarchicalClusteringP2P):
    """VGHierarchicalClusteringP2P with documents that have no label at the level being scored dropped.

    Same data and same revision as VGHierarchicalClusteringP2P. The only difference is that a
    document whose label path stops above the level being scored is left out of that
    level rather than gathered into one group under a sentinel label. Those documents
    have nothing in common except a missing label, so scoring them asks the model to
    find a class that is not really there.

    Scores are not comparable with VGHierarchicalClusteringP2P. On the SNL tasks the difference is
    about +0.13 v_measure, and it tracks the share of documents that have no label at
    each level.
    """

    drop_unlabelled_documents = True

    metadata = VGHierarchicalClusteringP2P.metadata.model_copy(
        update={"name": "VGHierarchicalClusteringP2P.v2", "superseded_by": None},
    )


class VGHierarchicalClusteringS2SV2(VGHierarchicalClusteringS2S):
    """VGHierarchicalClusteringS2S with documents that have no label at the level being scored dropped.

    Same data and same revision as VGHierarchicalClusteringS2S. The only difference is that a
    document whose label path stops above the level being scored is left out of that
    level rather than gathered into one group under a sentinel label. Those documents
    have nothing in common except a missing label, so scoring them asks the model to
    find a class that is not really there.

    Scores are not comparable with VGHierarchicalClusteringS2S. On the SNL tasks the difference is
    about +0.13 v_measure, and it tracks the share of documents that have no label at
    each level.
    """

    drop_unlabelled_documents = True

    metadata = VGHierarchicalClusteringS2S.metadata.model_copy(
        update={"name": "VGHierarchicalClusteringS2S.v2", "superseded_by": None},
    )
