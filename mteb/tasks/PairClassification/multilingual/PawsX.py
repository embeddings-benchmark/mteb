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
            "trust_remote_code": True,
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
        for lang in self.langs:
            hf_dataset = self.dataset[lang]

            # Rename columns
            hf_dataset = hf_dataset.rename_columns("sentence1", "sent1")
            hf_dataset = hf_dataset.rename_columns("sentence2", "sent2")
            hf_dataset = hf_dataset.rename_columns("label", "labels")

            self.dataset[lang] = hf_dataset
