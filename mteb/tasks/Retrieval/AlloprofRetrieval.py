import datasets

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlloprofRetrieval(AbsTaskRetrieval):

    @property
    def description(self):
        return {
            'name': 'AlloprofRetrieval',
            'hf_hub_name': 'lyon-nlp/alloprof',
            'reference': 'https://huggingface.co/datasets/antoinelb7/alloprof',
            "description": (
                "This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum"
                "curated by a large number of teachers to students on all subjects taught from in primary and secondary school"
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
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(self.description['hf_hub_name'], "documents")
        queries_raw = datasets.load_dataset(self.description['hf_hub_name'], "queries")

        self.queries = {
            "test": {
                str(q["id"]):q["text"] for q
                in queries_raw["queries"]
                }
            }
        
        self.corpus = {
            "test": {
                str(d["uuid"]):{"text":d["text"]} for d
                in corpus_raw["documents"]
                }
            }
        
        self.relevant_docs = {"test": {}}
        for q in queries_raw["queries"]:
            for r in q["relevant"]:
                self.relevant_docs["test"][str(q["id"])] = {r:1}

        self.data_loaded = True
