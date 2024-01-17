from ...abstasks.AbsTaskSTS import AbsTaskSTS


class STSES(AbsTaskSTS):
    @property
    def description(self):
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
