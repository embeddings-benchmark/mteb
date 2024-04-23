from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class MLSUMClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MLSUMClusteringP2P",
        description="Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.",
        reference="https://huggingface.co/datasets/mlsum",
        dataset={
            "path": "mlsum",
            "revision": "b5d54f8f3b61ae17845046286940f03c6bc79bc7",
            "name": "fr",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
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

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(
            self.metadata.dataset["path"],
            self.metadata.dataset["name"],
            split=self.metadata.eval_splits[0],
            revision=self.metadata.dataset["revision"],
        )
        self.dataset_transform()
        self.data_loaded = True

    def create_description(self, example):
        example["text"] = example["title"] + " " + example["text"]
        return example

    def dataset_transform(self):
        """Convert to standard format"""
        self.dataset = self.dataset.map(self.create_description)
        self.dataset = self.dataset.remove_columns(["summary", "url", "date", "title"])
        texts = self.dataset["text"]
        topics = self.dataset["topic"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(texts, 10)],
            "labels": [split.tolist() for split in np.array_split(topics, 10)],
        }
        self.dataset = {
            self.metadata_dict["eval_splits"][0]: datasets.Dataset.from_dict(new_format)
        }
