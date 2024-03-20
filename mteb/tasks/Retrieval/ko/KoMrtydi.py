from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoMrtydi(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "Ko-mrtydi",
            "hf_hub_name": "taeminlee/Ko-mrtydi",
            "description": "Ko-mrtydi",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["ko"],
            "main_score": "ndcg_at_10",
            "revision": "71a2e011a42823051a2b4eb303a3366bdbe048d3",
        }
