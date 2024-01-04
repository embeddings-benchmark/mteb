import datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlloprofRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["test"]

    @property
    def description(self):
        return {
            "name": "AlloprofRetrieval",
            "hf_hub_name": "lyon-nlp/alloprof",
            "reference": "https://huggingface.co/datasets/antoinelb7/alloprof",
            "description": (
                "This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum"
                "curated by a large number of teachers to students on all subjects taught from in primary and secondary school"
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": self._EVAL_SPLITS,
            "eval_langs": ["fr"],
            "main_score": "ndcg_at_10",
            "revision": "75e7c6bf9d618062c5b42ad6d06e10494d2b3abb",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(self.description["hf_hub_name"], "documents")
        queries_raw = datasets.load_dataset(self.description["hf_hub_name"], "queries")

        self.queries = {self._EVAL_SPLITS[0]: {str(q["id"]): q["text"] for q in queries_raw["queries"]}}
        self.corpus = {self._EVAL_SPLITS[0]: {str(d["uuid"]): {"text": d["text"]} for d in corpus_raw["documents"]}}

        self.relevant_docs = {self._EVAL_SPLITS[0]: {}}
        for q in queries_raw["queries"]:
            for r in q["relevant"]:
                self.relevant_docs[self._EVAL_SPLITS[0]][str(q["id"])] = {r: 1}

        self.data_loaded = True
