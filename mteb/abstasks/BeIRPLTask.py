import os
from .AbsTask import AbsTask

import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
beir_path = os.path.join(current_dir, "..", "..", "..", "beir")
sys.path.append(beir_path)


class BeIRPLTask(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR-PL benchmark.
        """
        try:
            from beir.datasets.data_loader_hf import HFDataLoader
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")


        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        dataset = self.description["beir_name"]

        # cqadupstack not on huggingface yet
        # dataset, sub_dataset = dataset.split("/") if "cqadupstack" in dataset else (dataset, None)

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        for split in eval_splits:

            corpus, queries, qrels = HFDataLoader(hf_repo=f"clarin-knext/{dataset}", streaming=False, keep_in_memory=False).load(split=split)
            # Conversion from DataSet
            queries = {query['id']: {'text': query['text']} for query in queries}
            corpus = {doc['id']: {'title': doc['title'] , 'text': doc['text']} for doc in corpus}

            self.corpus[split], self.queries[split], self.relevant_docs[split] = corpus, queries, qrels

        self.data_loaded = True
