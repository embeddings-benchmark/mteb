from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class MLSUMClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MLSUMClusteringS2S",
        description="Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.",
        reference="https://huggingface.co/datasets/mlsum",
        hf_hub_name="mteb/mlsum",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="v_measure",
        revision="b5d54f8f3b61ae17845046286940f03c6bc79bc7",
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
            "fr",
            split=self.metadata_dict["eval_splits"][0],
            revision=self.metadata_dict.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """
        Convert to standard format
        """
        self.dataset = self.dataset.remove_columns(["summary", "text", "url", "date"])
        new_format = {
            "sentences": [
                split.tolist() for split in np.array_split(self.dataset["title"], 10)
            ],
            "labels": [
                split.tolist() for split in np.array_split(self.dataset["topic"], 10)
            ],
        }
        self.dataset = {
            self.metadata_dict["eval_splits"][0]: datasets.Dataset.from_dict(new_format)
        }
