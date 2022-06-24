import os

import datasets

from beir import util


USE_BEIR_DEVELOPMENT = False
try:
    from beir.datasets.data_loader_hf import HFDataLoader as BeirDataLoader

    USE_BEIR_DEVELOPMENT = True
except ImportError:
    from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader

from .AbsTask import AbsTask


class BeIRTask(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        dataset = self.description["beir_name"]
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            if USE_BEIR_DEVELOPMENT:
                self.corpus[split], self.queries[split], self.relevant_docs[split] = BeirDataLoader(
                    hf_repo=f"BeIR/{dataset}"
                ).load(split=split)
            else:
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
                download_path = os.path.join(datasets.config.HF_DATASETS_CACHE, "BeIR")
                data_path = util.download_and_unzip(url, download_path)
                self.corpus[split], self.queries[split], self.relevant_docs[split] = BeirDataLoader(
                    data_folder=data_path
                ).load(split=split)
        self.data_loaded = True
