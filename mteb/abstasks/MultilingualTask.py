from abc import ABC, abstractmethod
import datasets
from .AbsTask import AbsTask

class MultilingualTask(AbsTask):
    def __init__(self, langs=None, **kwargs):
        super(MultilingualTask, self).__init__(**kwargs)
        self.langs = langs if langs else self.description["available_langs"]
        self.is_multilingual = True
        

    def load_data(self):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(self.description["hf_hub_name"], lang)
        self.data_loaded = True