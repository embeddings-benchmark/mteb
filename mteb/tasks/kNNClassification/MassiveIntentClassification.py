from ...abstasks import AbsTaskKNNClassification, MultilingualTask

_LANGUAGES = ['af', 'am', 'ar', 'az', 'bn', 'cy', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'km', 'kn', 'ko', 'lv', 'ml', 'mn', 'ms', 'my', 'nb', 'nl', 'pl', 'pt', 'ro', 'ru', 'sl', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh-CN', 'zh-TW']


class MassiveIntentClassification(MultilingualTask, AbsTaskKNNClassification):
    @property
    def description(self):
        return {
            "name": "MassiveIntentClassification",
            "hf_hub_name": "mteb/amazon_massive_intent",
            "description": "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages",
            "reference": "https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.",
            "category": "s2s",
            "type": "kNNClassification",
            "available_splits": ["train", "validation", "test"],
            "available_langs": _LANGUAGES,
            "main_score": "accuracy",
        }
