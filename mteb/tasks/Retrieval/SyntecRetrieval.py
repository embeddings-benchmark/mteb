import datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SyntecRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["test"]


    @property
    def description(self):
        return {
            "name": "SyntecRetrieval",
            "hf_hub_name": "lyon-nlp/mteb-fr-retrieval-syntec-s2p",
            "reference": "https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p",
            "description": (
                "This dataset has been built from the Syntec Collective bargaining agreement."
                "It maps a question to an article from the agreement"
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": self._EVAL_SPLITS,
            "eval_langs": ["fr"],
            "main_score": "ndcg_at_5",
            "revision": "77f7e271bf4a92b24fce5119f3486b583ca016ff",
        }


    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(self.description["hf_hub_name"], "documents")
        queries_raw = datasets.load_dataset(self.description["hf_hub_name"], "queries")

        self.queries = {
            self._EVAL_SPLITS[0]: {
                str(i): q["Question"] 
                for i, q in enumerate(queries_raw["queries"])}
            }
        self.corpus = {
            self._EVAL_SPLITS[0]: {
                str(d["id"]): {"text": d["title"] + "\n\n" + d["content"]} 
                for d in corpus_raw["documents"]
                }}

        self.relevant_docs = {
            self._EVAL_SPLITS[0]: {
                str(i) : {str(q["Article"]): 1}
                for i, q in enumerate(queries_raw["queries"])
        }}

        self.data_loaded = True
