from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PawsXMaltese(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsXMaltese",
        description="amitness/PAWS-X-maltese",
        reference="https://aclanthology.org/D19-1382/",
        dataset={
            "path": "amitness/PAWS-X-maltese",
            "revision": "632ced95db1e042e3487c9f8ea3ef4187f666299",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["mlt-Latn"],
        main_score="ap",
        date=("2023-05-03", "2023-05-03"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"train": 49175, "validation": 2000, "test": 2000},
        avg_character_length={"train": 114.21, "validation": 113.44, "test": 114.62},
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sent1": hf_dataset["sentence1_mt"],
                    "sent2": hf_dataset["sentence2_mt"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
