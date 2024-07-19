from __future__ import annotations

from mteb.abstasks import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast

N_SAMPLES = 2048


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(",")[:2]
    return record


class VGHierarchicalClusteringP2P(AbsTaskClusteringFast):
    max_document_to_embed = N_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="VGHierarchicalClusteringP2P",
        dataset={
            "path": "navjordj/VG_summarization",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
        },
        description="Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.",
        reference="https://huggingface.co/datasets/navjordj/VG_summarization",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="CC-BY-NC 4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation="""@mastersthesis{navjord2023beyond,
    title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
    author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
    year={2023},
    school={Norwegian University of Life Sciences, {\AA}s}
}""",
        descriptive_stats={
            "n_samples": {"test": N_SAMPLES},
            "avg_character_length": {"test": 2670.3243084794544},
        },
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"article": "sentences", "classes": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
        # Subsampling the dataset
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=N_SAMPLES, seed=self.seed
        )["test"]


class VGHierarchicalClusteringS2S(AbsTaskClusteringFast):
    max_document_to_embed = N_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="VGHierarchicalClusteringS2S",
        dataset={
            "path": "navjordj/VG_summarization",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
        },
        description="Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.",
        reference="https://huggingface.co/datasets/navjordj/VG_summarization",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="CC-BY-NC 4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation="""@mastersthesis{navjord2023beyond,
    title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
    author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
    year={2023},
    school={Norwegian University of Life Sciences, {\AA}s}
}""",
        descriptive_stats={
            "n_samples": {"test": N_SAMPLES},
            "avg_character_length": {"test": 139.31247668283325},
        },
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"ingress": "sentences", "classes": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
        # Subsampling the dataset
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=N_SAMPLES, seed=self.seed
        )["test"]
