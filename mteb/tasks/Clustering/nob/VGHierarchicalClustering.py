from __future__ import annotations

from mteb.abstasks import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(",")[:2]
    return record


class VGHierarchicalClusteringP2P(AbsTaskClusteringFast):
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
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        form=["written"],
        domains=["News", "Non-fiction"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        text_creation="found",
        bibtex_citation="""@mastersthesis{navjord2023beyond,
    title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
    author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
    year={2023},
    school={Norwegian University of Life Sciences, {\AA}s}
}""",
        n_samples={"test": 18763},
        avg_character_length={"test": 2670.3243084794544},
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"article": "sentences", "classes": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=2048,
        )
        print(self.dataset)


class VGHierarchicalClusteringS2S(AbsTaskClusteringFast):
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
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        form=["written"],
        domains=["News", "Non-fiction"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        text_creation="found",
        bibtex_citation="""@mastersthesis{navjord2023beyond,
    title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
    author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
    year={2023},
    school={Norwegian University of Life Sciences, {\AA}s}
}""",
        n_samples={"test": 18763},
        avg_character_length={"test": 139.31247668283325},
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"ingress": "sentences", "classes": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=2048,
        )
        print(self.dataset)
