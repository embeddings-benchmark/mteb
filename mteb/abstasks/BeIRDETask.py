import os
from .AbsTask import AbsTask
from .AbsTaskRetrieval import AbsTaskRetrieval


class BeIRDETask(AbsTask):
# TODO: beyond copying from BeIRPLTask this is unrelated, rename this
# class BeIRDETask(AbsTaskRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, eval_splits=None, **kwargs):
        """
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


        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        for split in eval_splits:

            # corpus, queries, qrels = HFDataLoader(hf_repo=f"deepset/{dataset}", streaming=False, keep_in_memory=False).load(split=split)
            corpus, queries, qrels = HFDataLoader(data_folder="scripts/data/germanquad", streaming=False, keep_in_memory=False).load(split=split)
            # Conversion from DataSet
            queries = {query['id']: {'text': query['text']} for query in queries}
            # corpus = {doc['id']: {'title': doc['title'] , 'text': doc['text']} for doc in corpus}
            corpus = {doc['id']: {'text': doc['text']} for doc in corpus}

            self.corpus[split], self.queries[split], self.relevant_docs[split] = corpus, queries, qrels

        self.data_loaded = True
