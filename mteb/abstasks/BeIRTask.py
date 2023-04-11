import os

import datasets

from .AbsTask import AbsTask


class BeIRTask(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        try:
            from beir import util
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        USE_BEIR_DEVELOPMENT = False
        try:
            raise ImportError("MTEB is temporarily incompatible with HFDataLoader")

            if self.description["beir_name"].startswith("cqadupstack"):
                raise ImportError("CQADupstack is incompatible with latest BEIR")
            from beir.datasets.data_loader_hf import HFDataLoader as BeirDataLoader

            USE_BEIR_DEVELOPMENT = True
        except ImportError:
            from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader

        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        dataset = self.description["beir_name"]
        dataset, sub_dataset = dataset.split("/") if "cqadupstack" in dataset else (dataset, None)

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
                data_path = f"{data_path}/{sub_dataset}" if sub_dataset else data_path
                self.corpus[split], self.queries[split], self.relevant_docs[split] = BeirDataLoader(
                    data_folder=data_path
                ).load(split=split)
        self.data_loaded = True
