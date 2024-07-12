from __future__ import annotations

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(",")
    # First level is trivial
    record["labels"] = record["labels"][1:]
    return record


class SNLHierarchicalClusteringP2P(AbsTaskClusteringFast):
    max_document_to_embed = 1300
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="SNLHierarchicalClusteringP2P",
        dataset={
            "path": "navjordj/SNL_summarization",
            "revision": "3d3d27aa7af8941408cefc3991ada5d12a4273d1",
        },
        description="Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
        reference="https://huggingface.co/datasets/navjordj/SNL_summarization",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="Not specified",
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
            "n_samples": {"test": 1300},
            "avg_character_length": {"test": 1986.9453846153847},
        },
    )
    max_depth = 5

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"article": "sentences", "category": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)


class SNLHierarchicalClusteringS2S(AbsTaskClusteringFast):
    max_document_to_embed = 1300
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="SNLHierarchicalClusteringS2S",
        dataset={
            "path": "navjordj/SNL_summarization",
            "revision": "3d3d27aa7af8941408cefc3991ada5d12a4273d1",
        },
        description="Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
        reference="https://huggingface.co/datasets/navjordj/SNL_summarization",
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="Not specified",
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
            "n_samples": {"test": 1300},
            "avg_character_length": {"test": 242.22384615384615},
        },
    )
    max_depth = 5

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"ingress": "sentences", "category": "labels"}
        )
        self.dataset = self.dataset.map(split_labels)
