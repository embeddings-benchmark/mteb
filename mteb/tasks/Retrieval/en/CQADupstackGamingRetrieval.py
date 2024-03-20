from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackGamingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CQADupstackGamingRetrieval",
            "hf_hub_name": "mteb/cqadupstack-gaming",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "4885aa143210c98657558c04aaf3dc47cfb54340",
        }
