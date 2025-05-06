from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AlloProfClusteringP2P(AbsTaskClustering):
    superseded_by = "AlloProfClusteringP2P.v2"

    metadata = TaskMetadata(
        name="AlloProfClusteringP2P",
        description="Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "lyon-nlp/alloprof",
            "revision": "392ba3f5bcc8c51f578786c1fc3dae648662cb9b",
            "name": "documents",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        date=("1996-01-01", "2023-04-14"),
        form=None,
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@misc{lef23,
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  doi = {10.48550/ARXIV.2302.07738},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  url = {https://arxiv.org/abs/2302.07738},
  year = {2023},
}
""",
    )

    def create_description(self, example):
        example["text"] = example["title"] + " " + example["text"]
        return example

    def dataset_transform(self):
        """Convert to standard format"""
        self.dataset = self.dataset.remove_columns("uuid")
        self.dataset = self.dataset.map(self.create_description)
        texts = self.dataset["documents"]["text"]
        topics = self.dataset["documents"]["topic"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(texts, 10)],
            "labels": [split.tolist() for split in np.array_split(topics, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)
        self.dataset.pop("documents")


class AlloProfClusteringP2PFast(AbsTaskClusteringFast):
    max_document_to_embed = 2556
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="AlloProfClusteringP2P.v2",
        description="Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "lyon-nlp/alloprof",
            "revision": "392ba3f5bcc8c51f578786c1fc3dae648662cb9b",
            "name": "documents",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        # (date of founding of the dataset source site, date of dataset paper publication)
        date=("1996-01-01", "2023-04-14"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lef23,
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  doi = {10.48550/ARXIV.2302.07738},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  url = {https://arxiv.org/abs/2302.07738},
  year = {2023},
}
""",
        adapted_from=["AlloProfClusteringP2P"],
    )

    def create_description(self, example):
        example["sentences"] = example["title"] + " " + example["text"]
        return example

    def dataset_transform(self):
        self.dataset["test"] = (
            self.dataset["documents"]
            .map(self.create_description)
            .rename_columns({"topic": "labels"})
            .select_columns(["sentences", "labels"])
        )
        self.dataset.pop("documents")
        unique_labels = list(set(self.dataset["test"]["labels"]))
        unique_labels.sort()
        self.dataset["test"] = self.dataset["test"].cast(
            datasets.Features(
                sentences=datasets.Value("string"),
                labels=datasets.ClassLabel(names=unique_labels),
            )
        )
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])
