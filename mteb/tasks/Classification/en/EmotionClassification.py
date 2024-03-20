from ....abstasks import AbsTaskClassification


class EmotionClassification(AbsTaskClassification):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "EmotionClassification",
            "hf_hub_name": "mteb/emotion",
            "description": (
                "Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love,"
                " sadness, and surprise. For more detailed information please refer to the paper."
            ),
            "reference": "https://www.aclweb.org/anthology/D18-1404",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "4f58c6b202a23cf9a4da393831edf4f9183cad37",
        }
