from __future__ import annotations

import random
from abc import ABC, abstractmethod

import datasets
import numpy as np
import torch

from mteb.abstasks.TaskMetadata import TaskMetadata


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

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision", None),
        )
        self.data_loaded = True

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

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
