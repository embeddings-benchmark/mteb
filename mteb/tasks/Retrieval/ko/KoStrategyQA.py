from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoStrategyQA(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "Ko-StrategyQA",
            "hf_hub_name": "taeminlee/Ko-StrategyQA",
            "description": "Ko-StrategyQA",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["ko"],
            "main_score": "ndcg_at_10",
            "revision": "d243889a3eb6654029dbd7e7f9319ae31d58f97c",
        }
