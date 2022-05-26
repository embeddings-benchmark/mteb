from ...abstasks import AbsTaskKNNClassification


class ToxicConversationsClassification(AbsTaskKNNClassification):
    @property
    def description(self):
        return {
            "name": "ToxicConversationsClassification",
            "hf_hub_name": "mteb/toxic_conversations_50k",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview",
            "category": "s2s",
            "type": "kNNClassification",
            "available_splits": ["train", "test"],
            "available_langs": ["en"],
            "main_score": "accuracy",
        }
