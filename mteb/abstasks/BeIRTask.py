from abc import ABC, abstractmethod
import datasets
from .AbsTask import AbsTask
from beir import util
from beir.datasets.data_loader import GenericDataLoader

class BeIRTask(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        if self.data_loaded:
            return
        dataset = self.description["beir_name"]
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, "datasets")
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in self.description["eval_splits"]:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(data_folder=data_path).load(split=split)
        self.data_loaded = True