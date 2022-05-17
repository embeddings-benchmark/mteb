from ...abstasks import AbsTaskKNNClassification


class Banking77Classification(AbsTaskKNNClassification):
    @property
    def description(self):
        return {
            "name": "Banking77Classification",
            "hf_hub_name": "mteb/banking77",
            "description": "Dataset composed of online banking queries annotated with their corresponding intents.",
            "reference": "https://arxiv.org/abs/2003.04807",
            "category": "s2s",
            "type": "kNNClassification",
            "available_splits": ["train", "test"],
            "available_langs": ["en"],
            "main_score": "accuracy",
        }
