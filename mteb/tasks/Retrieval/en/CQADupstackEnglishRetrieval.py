from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackEnglishRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CQADupstackEnglishRetrieval",
            "hf_hub_name": "mteb/cqadupstack-english",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "ad9991cb51e31e31e430383c75ffb2885547b5f0",
        }
