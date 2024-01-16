import datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class XMarketES(AbsTaskRetrieval):
    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            "name": "XMarketES",
            "hf_hub_name": "jinaai/xmarket_ml",
            "description": "XMarket is an ecommerce category to product retrieval dataset in Spanish.",
            "reference": "https://xmrec.github.io/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "ndcg_at_10",
            "revision": "705db869e8107dfe6e34b832af90446e77d813e3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(
            self.description["hf_hub_name"],
            "queries",
            languages=['es'],
            split=self._EVAL_SPLIT,
            trust_remote_code=True,
        )
        corpus_rows = datasets.load_dataset(
            self.description["hf_hub_name"], "corpus", languages=['es'], split=self._EVAL_SPLIT, trust_remote_code=True
        )
        qrels_rows = datasets.load_dataset(
            self.description["hf_hub_name"], "qrels", languages=['es'], split=self._EVAL_SPLIT, trust_remote_code=True
        )

        self.queries = {self._EVAL_SPLIT: {row["_id"]: row["text"] for row in query_rows}}
        self.corpus = {self._EVAL_SPLIT: {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows}
        }

        self.data_loaded = True
