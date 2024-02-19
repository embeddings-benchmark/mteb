import datasets
import numpy as np

from ...abstasks.AbsTaskClustering import AbsTaskClustering


class MasakhaNEWSClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "MasakhaNEWSClusteringS2S",
            "hf_hub_name": "masakhane/masakhanews",
            "description": (
                "Clustering of news article headlines from MasakhaNEWS dataset. Clustering of 10 sets on the news article label."
            ),
            "reference": "https://huggingface.co/datasets/masakhane/masakhanews",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "v_measure",
            "revision": "8ccc72e69e65f40c70e117d8b3c08306bb788b60",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            "fra",
            revision=self.description.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """
        Convert to standard format
        """
        self.dataset = self.dataset.remove_columns(["url", "text", "headline_text"])
        headlines = (
            self.dataset["train"]["headline"]
            + self.dataset["validation"]["headline"]
            + self.dataset["test"]["headline"]
        )
        labels = self.dataset["train"]["label"] + self.dataset["validation"]["label"] + self.dataset["test"]["label"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(headlines, 10)],
            "labels": [split.tolist() for split in np.array_split(labels, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)
        self.dataset.pop("train")
        self.dataset.pop("validation")
