import datasets
import numpy as np

from ...abstasks.AbsTaskClustering import AbsTaskClustering


class AlloProfClusteringS2S(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "AlloProfClusteringS2S",
            "hf_hub_name": "lyon-nlp/alloprof",
            "description": (
                "Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic."
            ),
            "reference": "https://huggingface.co/datasets/lyon-nlp/alloprof",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "v_measure",
            "revision": "3e394f99549416c102c44af9b38460d0d07fa10f",
        }
    
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            "documents",
            revision=self.description.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """
        Convert to standard format
        """
        self.dataset = self.dataset.remove_columns("uuid")
        self.dataset = self.dataset.remove_columns("text")
        titles = self.dataset["documents"]["title"]
        topics = self.dataset["documents"]["topic"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(titles, 10)],
            "labels": [split.tolist() for split in np.array_split(topics, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)
        self.dataset.pop("documents")
