from datasets import load_dataset

from ....abstasks.AbsTaskSTS import AbsTaskSTS

_EVAL_SPLIT = "test"


class STSES(AbsTaskSTS):
    @property
    def metadata_dict(self):
        return {
            "name": "STSES",
            "hf_hub_name": "PlanTL-GOB-ES/sts-es",
            "description": "Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)",
            "reference": "https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "0912bb6c9393c76d62a7c5ee81c4c817ff47c9f4",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = load_dataset(
            self.metadata_dict["hf_hub_name"],
            trust_remote_code=True,
            revision=self.metadata_dict.get("revision", None),
        )[_EVAL_SPLIT]
        data = data.add_column("score", [d["label"] for d in data])
        self.dataset = {_EVAL_SPLIT: data}

        self.data_loaded = True
