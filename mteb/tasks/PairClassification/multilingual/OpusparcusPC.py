from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskPairClassification, MultilingualTask

_LANGUAGES = ["de", "en", "fi", "fr", "ru", "sv"]


class OpusparcusPC(AbsTaskPairClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="OpusparcusPC",
        hf_hub_name="GEM/opusparcus",
        description="Opusparcus is a paraphrase corpus for six European language: German, English, Finnish, French, Russian, and Swedish. The paraphrases consist of subtitles from movies and TV shows.",
        reference="https://gem-benchmark.com/data_cards/opusparcus",
        category="s2s",
        type="PairClassification",
        eval_splits=["test.full", "validation.full"],
        eval_langs=_LANGUAGES,
        main_score="ap",
        revision="9e9b1f8ef51616073f47f306f7f47dd91663f86a",
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

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                self.metadata_dict["hf_hub_name"],
                lang=lang,
                quality=100,
                revision=self.metadata_dict.get("revision", None),
            )
            self.dataset_transform(lang)
        self.data_loaded = True

    def dataset_transform(self, lang):
        for split in self.dataset[lang]:
            # Renaming features
            labels = self.dataset[lang][split]["annot_score"]
            sent1 = self.dataset[lang][split]["input"]
            sent2 = self.dataset[lang][split]["target"]
            new_dict = {}
            # Labels are a score between 1.0 and 4.0, and we need binary classification
            labels = [
                0 if label < 2.5 else 1 if label > 2.5 else 2.5 for label in labels
            ]
            # Get neutral label to delete them
            neutral = [i for i, val in enumerate(labels) if val == 2.5]
            for i in sorted(neutral, reverse=True):
                del labels[i]
                del sent1[i]
                del sent2[i]
            new_dict["labels"] = [labels]
            new_dict["sent1"] = [sent1]
            new_dict["sent2"] = [sent2]
            self.dataset[lang][split] = datasets.Dataset.from_dict(new_dict)
