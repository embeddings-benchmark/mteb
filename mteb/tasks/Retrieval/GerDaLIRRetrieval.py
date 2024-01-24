from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class SciFact(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "GerDaLIR",
            "hf_hub_name": "jinaai/gerdalir",
            "description": (
                "GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform."
            ),
            "reference": "https://github.com/lavis-nlp/GerDaLIR",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(self.description["hf_hub_name"], "queries", split=self._EVAL_SPLIT)
        corpus_rows = datasets.load_dataset(self.description["hf_hub_name"], "corpus", split=self._EVAL_SPLIT)
        qrels_rows = datasets.load_dataset(self.description["hf_hub_name"], "qrels", split=self._EVAL_SPLIT)

        self.queries = {self._EVAL_SPLIT: {row["_id"]: row["text"] for row in query_rows}}
        self.corpus = {self._EVAL_SPLIT: {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows}
        }

        self.data_loaded = True
