from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class HALClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="HALClusteringS2S",
        description="Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)",
        reference="https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s",
        hf_hub_name="lyon-nlp/clustering-hal-s2s",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="v_measure",
        revision="e06ebbbb123f8144bef1a5d18796f3dec9ae2915",
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
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """
        Convert to standard format
        """
        self.dataset = self.dataset.remove_columns("hal_id")
        titles = self.dataset["test"]["title"]
        domains = self.dataset["test"]["domain"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(titles, 10)],
            "labels": [split.tolist() for split in np.array_split(domains, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)
