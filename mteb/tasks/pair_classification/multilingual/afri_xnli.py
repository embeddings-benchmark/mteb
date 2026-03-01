from __future__ import annotations

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AfriXNLI(AbsTaskPairClassification):
    """Custom task for the AfriXNLI dataset."""

    metadata = TaskMetadata(
        name="AfriXNLI",
        description=(
            "Cross-lingual natural language inference dataset focusing on "
            "African languages."
        ),
        reference="https://github.com/masakhane-io/afri-xnli",
        dataset={
            "path": "masakhane/afrixnli",  # dataset on the HF hub
            "revision": "e3ca06b30f3e7af2a86f6c8609ea76fee326bc56",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "eng": ["eng-Latn"],
            "fra": ["fra-Latn"],
            "amh": ["amh-Ethi"],
            "ewe": ["ewe-Latn"],
            "hau": ["hau-Latn"],
            "ibo": ["ibo-Latn"],
            "kin": ["kin-Latn"],
            "lin": ["lin-Latn"],
            "lug": ["lug-Latn"],
            "gaz": ["orm-Ethi"],
            "sna": ["sna-Latn"],
            "sot": ["sot-Latn"],
            "swh": ["swa-Latn"],
            "twi": ["twi-Latn"],
            "wol": ["wol-Latn"],
            "xho": ["xho-Latn"],
            "yor": ["yor-Latn"],
            "zul": ["zul-Latn"],
        },
        main_score="max_ap",
        date=("2020-01-01", "2020-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
    )

    def dataset_transform(self, **kwargs):
        for lang in self.dataset:
            for split in self.dataset[lang]:
                # keep only entail (0) / contradict (2)
                ds = self.dataset[lang][split].filter(lambda x: x["label"] in (0, 2))

                # map to binary labels and standard column names
                def map_labels(example):
                    return {
                        "sentence1": example["premise"],
                        "sentence2": example["hypothesis"],
                        "labels": 0 if example["label"] == 2 else 1,
                    }

                self.dataset[lang][split] = ds.map(map_labels, remove_columns=ds.column_names)
