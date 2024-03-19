from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoMiracl(AbsTaskRetrieval):
    @property
    def metadata_dict(self):
        return {
            "name": "Ko-miracl",
            "hf_hub_name": "taeminlee/Ko-miracl",
            "description": "Ko-miracl",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["ko"],
            "main_score": "ndcg_at_10",
            "revision": "5c7690518e481375551916f24241048cf7b017d0",
        }
