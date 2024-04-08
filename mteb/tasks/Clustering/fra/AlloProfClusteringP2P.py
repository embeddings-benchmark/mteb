from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class AlloProfClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="AlloProfClusteringP2P",
        description="Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "lyon-nlp/alloprof",
            "revision": "392ba3f5bcc8c51f578786c1fc3dae648662cb9b",
            "name": "documents",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def create_description(self, example):
        example["text"] = example["title"] + " " + example["text"]
        return example

    def dataset_transform(self):
        """
        Convert to standard format
        """
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
