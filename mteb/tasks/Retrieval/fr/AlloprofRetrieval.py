import datasets

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlloprofRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
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
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "ndcg_at_10",
            "revision": "392ba3f5bcc8c51f578786c1fc3dae648662cb9b",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"], "documents"
        )
        queries_raw = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"], "queries"
        )
        eval_split = self.metadata_dict["eval_splits"][0]
        self.queries = {
            eval_split: {str(q["id"]): q["text"] for q in queries_raw[eval_split]}
        }
        self.corpus = {
            eval_split: {
                str(d["uuid"]): {"text": d["text"]} for d in corpus_raw["documents"]
            }
        }

        self.relevant_docs = {eval_split: {}}
        for q in queries_raw[eval_split]:
            for r in q["relevant"]:
                self.relevant_docs[eval_split][str(q["id"])] = {r: 1}

        self.data_loaded = True
