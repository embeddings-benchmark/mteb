from ...abstasks.AbsTaskKNNClassification import AbsTaskKNNClassification

_LANGUAGES = []


class Banking77Classification(AbsTaskKNNClassification):
    def __init__(self, available_langs=None):
        super().__init__()
        self.available_langs = available_langs if available_langs else _LANGUAGES

    @property
    def description(self):
        return {
            "name": "Banking77Classification",
            "hf_hub_name": "banking77",
            "description": "Dataset composed of online banking queries annotated with their corresponding intents.",
            "reference": "https://arxiv.org/abs/2003.04807",
            "category": "s2s",
            "type": "kNNClassification",
            "available_splits": ["train", "test"],
            "available_langs": self.available_langs,
            "main_score": "accuracy",
        }
