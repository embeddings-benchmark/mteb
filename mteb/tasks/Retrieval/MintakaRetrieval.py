from datasets import load_dataset

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.MultilingualTask import MultilingualTask


_EVAL_SPLIT = "test"
_LANGUAGES = ["ar", "de", "es", "fr", "hi", "it", "ja", "pt"]

class MintakaRetrieval(AbsTaskRetrieval, MultilingualTask):
    @property
    def description(self):
        return {
            "name": "MintakaRetrieval",
            "hf_hub_name": "jinaai/mintakaqa",
            "reference": "https://github.com/amazon-science/mintaka",
            "description": (
                "Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering."
            ),
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": _LANGUAGES,
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = load_dataset(
            self.description["hf_hub_name"],
            "es",
        )[_EVAL_SPLIT]

        queries = {}
        corpus = {}
        relevant_docs = {}

        for row in data:
            row_id = row["id"]
            question = row["question"]
            answer = row["answer"]
            query_id = f"Q{row_id}"
            queries[query_id] = question
            doc_id = f"D{row_id}"
            corpus[doc_id] = answer
            if query_id not in relevant_docs:
                relevant_docs[query_id] = {}
            relevant_docs[query_id][doc_id] = 1

        self.queries = {_EVAL_SPLIT: queries}
        self.corpus = {_EVAL_SPLIT: {k: {"text": v} for k, v in corpus.items()}}
        self.relevant_docs = {_EVAL_SPLIT: relevant_docs}

        self.data_loaded = True
