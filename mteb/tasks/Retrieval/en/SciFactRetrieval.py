from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFact(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "SciFact",
            "hf_hub_name": "mteb/scifact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "reference": "https://github.com/allenai/scifact",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["train", "test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "0228b52cf27578f30900b9e5271d331663a030d7",
        }
