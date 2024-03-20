from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackMathematicaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CQADupstackMathematicaRetrieval",
            "hf_hub_name": "mteb/cqadupstack-mathematica",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "90fceea13679c63fe563ded68f3b6f06e50061de",
        }
