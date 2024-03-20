import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PawsX(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsX",
        hf_hub_name="paws-x",
        description="",
        reference="https://arxiv.org/abs/1908.11828",
        category="s2s",
        type="PairClassification",
        eval_splits=["test.full", "validation.full"],
        eval_langs=["de", "en", "es", "fr", "ja", "ko", "zh"],
        main_score="ap",
        revision="8a04d940a42cd40658986fdd8e3da561533a3646",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.dataset = dict()
        for lang in self.langs:
            hf_dataset = datasets.load_dataset(
                self.metadata_dict["hf_hub_name"],
                lang,
                revision=self.metadata_dict.get("revision", None),
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
