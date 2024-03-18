import datasets

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PawsX(MultilingualTask, AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "PawsX",
            "hf_hub_name": "paws-x",
            "description": "",
            "reference": "",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test"],
            "eval_langs": ["de", "en", "es", "fr", "ja", "ko", "zh"],
            "main_score": "ap",
            "revision": "8a04d940a42cd40658986fdd8e3da561533a3646",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.dataset = dict()
        for lang in self.langs:
            hf_dataset = datasets.load_dataset(
                self.description["hf_hub_name"],
                lang,
                revision=self.description.get("revision", None),
            )

            sent1 = []
            sent2 = []
            labels = []

            for line in hf_dataset["test"]:
                sent1.append(line["sentence1"])
                sent2.append(line["sentence2"])
                labels.append(line["label"])

            self.dataset[lang] = {
                "test": [
                    {
                        "sent1": sent1,
                        "sent2": sent2,
                        "labels": labels,
                    }
                ]
            }

        self.data_loaded = True
