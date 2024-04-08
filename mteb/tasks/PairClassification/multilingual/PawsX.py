from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PawsX(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsX",
        dataset={
            "path": "paws-x",
            "revision": "8a04d940a42cd40658986fdd8e3da561533a3646",
        },
        description="",
        reference="https://arxiv.org/abs/1908.11828",
        category="s2s",
        type="PairClassification",
        eval_splits=["test", "validation"],
        eval_langs=["de", "en", "es", "fr", "ja", "ko", "zh"],
        main_score="ap",
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
        n_samples=None,
        avg_character_length=None,
    )

    def dataset_transform(self):
        _dataset = {}
        for lang in self.langs:
            _dataset[lang] = {}
            for split in self.metadata.eval_splits:
                hf_dataset = self.dataset[lang][split]

                _dataset[lang][split] = [
                    {
                        "sent1": hf_dataset["sentence1"],
                        "sent2": hf_dataset["sentence2"],
                        "labels": hf_dataset["label"],
                    }
                ]
        self.dataset = _dataset
