from __future__ import annotations

import random
from abc import ABC, abstractmethod

import datasets
import numpy as np
import torch

from mteb.abstasks.TaskMetadata import TaskMetadata

SEED = 42

class AbsTask(ABC):
    metadata: TaskMetadata

    def __init__(self, seed=SEED, **kwargs):
        self.dataset = None
        self.data_loaded = False
        self.is_multilingual = False
        self.is_crosslingual = False
        self.save_suffix = kwargs.get("save_suffix", "")

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def dataset_transform(self):
        """Transform operations applied to the dataset after loading.
        Override this method if your dataset requires any transformation.
        """
        pass

    @staticmethod
    def stratified_subsampling(
        dataset_dict: datasets.DatasetDict,
        splits: list[str] = ["test"],
        label: str = "label",
        n_samples: int = 2048,
    ) -> datasets.DatasetDict:
        """Subsamples the dataset with stratification by the supplied label.
        Returns a datasetDict object.
        :param splits: the splits of the dataset.
        :param label: the label with which the stratified sampling is based on.
        :param n_samples: Optional, number of samples to subsample. Default is max_n_samples.
        """

        ## Can only do this if the label column is of ClassLabel.
        if not isinstance(dataset_dict[splits[0]].features[label], datasets.ClassLabel):
            dataset_dict = dataset_dict.class_encode_column(label)

        ds_dict = {}
        for split in splits:
            ds_dict.update({split: dataset_dict[split].train_test_split(
                test_size=n_samples, seed=SEED, stratify_by_column=label
            )["test"]})  ## only take the specified test split.
        return datasets.DatasetDict(ds_dict)

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()
        self.data_loaded = True

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        return metadata_dict

    @abstractmethod
    def evaluate(self, model, split="test"):
        """Evaluates a Sentence Embedding Model on the task.
        Returns a dict (that can be serialized to json).
        :param model: Sentence embedding method. Implements a encode(sentences) method, that encodes sentences
        and returns a numpy matrix with the sentence embeddings
        :param split: Which datasplit to be used.
        """
        raise NotImplementedError

    @property
    def languages(self) -> set[str]:
        """Returns the languages of the task"""
        return self.metadata.languages

    def __repr__(self) -> str:
        """Format the representation of the task such that it appears as:

        TaskObjectName(name='{name}', languages={lang1, lang2, ...})
        """
        langs = self.languages
        if len(langs) > 3:
            langs = list(langs)[:3]
            langs.append("...")
        return (
            f"{self.__class__.__name__}(name='{self.metadata.name}', languages={langs})"
        )
