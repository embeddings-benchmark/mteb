import datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BSARDRetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'BSARDRetrieval',
            'hf_hub_name': 'maastrichtlawtech/bsard',
            'reference': 'https://huggingface.co/datasets/maastrichtlawtech/bsard',
            "description": (
                "The Belgian Statutory Article Retrieval Dataset (BSARD)"
                "is a French native dataset for studying legal information retrieval."
                "BSARD consists of more than 22,600 statutory articles from Belgian law"
                "and about 1,100 legal questions posed by Belgian citizens and labeled"
                "by experienced jurists with relevant articles from the corpus."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "ndcg_at_10",
        }


    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            self.description['hf_hub_name'], "corpus", split="corpus"
            )
        queries_raw = datasets.load_dataset(
            self.description['hf_hub_name'], "questions", split=self._EVAL_SPLIT
            )

        self.queries = {
            self._EVAL_SPLIT: {
                str(q["id"]): " ".join((q["question"] + q["extra_description"]))
                for q in queries_raw
                }
            }

        self.corpus = {
            self._EVAL_SPLIT: {
                str(d["id"]):{"text":d["article"]}
                for d in corpus_raw
                }
            }

        self.relevant_docs = {self._EVAL_SPLIT: {}}
        for q in queries_raw:
            for doc_id in q["article_ids"]:
                self.relevant_docs[self._EVAL_SPLIT][str(q["id"])] = {str(doc_id): 1}

        self.data_loaded = True
