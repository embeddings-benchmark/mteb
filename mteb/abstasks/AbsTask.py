from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

import datasets
import numpy as np
import torch

from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


def _log_dataset_configuration_deprecation_warning():
    logger.warning(
        "hf_hub_name and revision are deprecated. This will be removed in a future version. "
        "Use the dataset key instead, which can contains any argument passed to datasets.load_dataset. "
        "Refer to https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset"
    )


class AbsTask(ABC):
    metadata: TaskMetadata

    def __init__(self, seed=42, **kwargs):
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
        """
        Transform operations applied to the dataset after loading.
        Override this method if your dataset requires any transformation.
        """
        pass

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        if "dataset" in self.metadata_dict:
            self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        else:
            _log_dataset_configuration_deprecation_warning()
            self.dataset = datasets.load_dataset(
                self.metadata_dict["hf_hub_name"],
                revision=self.metadata_dict.get("revision", None),
            )
        self.dataset_transform()
        self.data_loaded = True

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        if "hf_hub_name" in metadata_dict:
            _log_dataset_configuration_deprecation_warning()
            metadata_dict["dataset"] = {
                "path": metadata_dict.pop("hf_hub_name"),
                "revision": metadata_dict.pop("revision", None),
            }
        return metadata_dict

    @abstractmethod
    def evaluate(self, model, split="test"):
        """
        Evaluates a Sentence Embedding Model on the task.
        Returns a dict (that can be serialized to json).
        :param model: Sentence embedding method. Implements a encode(sentences) method, that encodes sentences
        and returns a numpy matrix with the sentence embeddings
        :param split: Which datasplit to be used.
        """
        raise NotImplementedError
