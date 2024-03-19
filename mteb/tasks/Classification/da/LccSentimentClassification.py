from mteb.abstasks import AbsTaskClassification


class LccSentimentClassification(AbsTaskClassification):
    @property
    def metadata_dict(self):
        return {
            "name": "LccSentimentClassification",
            "hf_hub_name": "DDSC/lcc",
            "description": "The leipzig corpora collection, annotated for sentiment",
            "reference": "https://github.com/fnielsen/lcc-sentiment",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "de7ba3406ee55ea2cc52a0a41408fa6aede6d3c6",
        }
