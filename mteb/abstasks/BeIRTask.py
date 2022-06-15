import os

import datasets

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader

from .AbsTask import AbsTask


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
        download_path = os.path.join(datasets.config.HF_DATASETS_CACHE, "BeIR")
        data_path = util.download_and_unzip(url, download_path)
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in self.description["eval_splits"]:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = HFDataLoader(
                data_folder=data_path
            ).load(split=split)
        self.data_loaded = True
