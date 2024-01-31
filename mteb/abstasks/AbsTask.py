import logging
import random
from abc import ABC, abstractmethod

import datasets
import numpy as np
import torch


logger = logging.getLogger(__name__)


class AbsTask(ABC):
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
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    @property
    @abstractmethod
    def description(self):
        """
        Returns a description of the task. Should contain the following fields:
        name: Name of the task (usually equal to the class name. Should be a valid name for a path on disc)
        description: Longer description & references for the task
        type: Of the set: [sts]
        eval_splits: Splits used for evaluation as list, e.g. ['dev', 'test']
        main_score: Main score value for task
        """
        raise NotImplementedError

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

    def get_language(self):
        """ Return the first language of the task.
        This is not meaningful for multilingual or cross-lingual tasks.
        Also, a few Nordic tasks although not marked as multilingual, contain multiple languages.
        For them, the first language will be returned.
        """
        langs = self.description['eval_langs']
        if len(langs) != 1:
            name = self.description['name']
            logger.warning(f"For task {name}, the number of languages is not 1: {langs}. Is it multi- or cross-lingual?")
        return langs[0]
