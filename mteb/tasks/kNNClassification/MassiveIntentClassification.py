from ...abstasks.AbsTaskKNNClassification import AbsTaskKNNClassification
import datasets

_LANGUAGES = [
    "af-ZA",
    "am-ET",
    "ar-SA",
    "az-AZ",
    "bn-BD",
    "cy-GB",
    "da-DK",
    "de-DE",
    "el-GR",
    "en-US",
    "es-ES",
    "fa-IR",
    "fi-FI",
    "fr-FR",
    "he-IL",
    "hi-IN",
    "hu-HU",
    "hy-AM",
    "id-ID",
    "is-IS",
    "it-IT",
    "ja-JP",
    "jv-ID",
    "ka-GE",
    "km-KH",
    "kn-IN",
    "ko-KR",
    "lv-LV",
    "ml-IN",
    "mn-MN",
    "ms-MY",
    "my-MM",
    "nb-NO",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sl-SL",
    "sq-AL",
    "sv-SE",
    "sw-KE",
    "ta-IN",
    "te-IN",
    "th-TH",
    "tl-PH",
    "tr-TR",
    "ur-PK",
    "vi-VN",
    "zh-CN",
    "zh-TW",
]


class MassiveIntentClassification(AbsTaskKNNClassification):
    def __init__(self, available_langs=None):
        super().__init__()
        self.available_langs = available_langs if available_langs else _LANGUAGES
        self.is_multilingual = True

    def load_data(self):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.available_langs:
            self.dataset[lang] = datasets.load_dataset(self.description["hf_hub_name"], lang)

        self.data_loaded = True

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
            "available_langs": self.available_langs,
            "main_score": "accuracy",
        }
