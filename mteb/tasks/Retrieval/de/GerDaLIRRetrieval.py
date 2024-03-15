import datasets

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GerDaLIR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    @property
    def description(self):
        return {
            "name": "GerDaLIR",
            "hf_hub_name": "jinaai/ger_da_lir",
            "description": (
                "GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform."
            ),
            "reference": "https://github.com/lavis-nlp/GerDaLIR",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "ndcg_at_10",
            "revision": "0bb47f1d73827e96964edb84dfe552f62f4fd5eb",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(
            self.description["hf_hub_name"],
            "queries",
            revision=self.description.get("revision", None),
            split=self._EVAL_SPLIT,
        )
        corpus_rows = datasets.load_dataset(
            self.description["hf_hub_name"],
            "corpus",
            revision=self.description.get("revision", None),
            split=self._EVAL_SPLIT,
        )
        qrels_rows = datasets.load_dataset(
            self.description["hf_hub_name"],
            "qrels",
            revision=self.description.get("revision", None),
            split=self._EVAL_SPLIT,
        )

        self.queries = {self._EVAL_SPLIT: {row["_id"]: row["text"] for row in query_rows}}
        self.corpus = {self._EVAL_SPLIT: {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows}
        }

        self.data_loaded = True
