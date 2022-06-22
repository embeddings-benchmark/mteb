from abc import ABC, abstractmethod

import datasets

from .AbsTask import AbsTask


class CrosslingualTask(AbsTask):
    def __init__(self, langs=None, **kwargs):
        super().__init__(**kwargs)
        self.langs = langs if langs else self.description["eval_langs"]
        self.is_crosslingual = True

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(self.description["hf_hub_name"], lang)
        self.data_loaded = True
