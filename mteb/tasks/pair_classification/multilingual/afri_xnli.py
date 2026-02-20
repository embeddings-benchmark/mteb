from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class AfriXNLI(MultilingualTask, AbsTaskPairClassification):
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
            "revision": "main",
            "trust_remote_code": True,
        },
        type="PairClassification",
        category="s2s",
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
            "orm": ["orm-Ethi"],
            "sna": ["sna-Latn"],
            "sot": ["sot-Latn"],
            "swa": ["swa-Latn"],
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

    def dataset_transform(self):
        new_dataset = {}

        for lang in self.dataset:
            new_dataset[lang] = {}
            for split in self.dataset[lang]:
                ds = (
                    self.dataset[lang][split]
                    .filter(lambda x: x["label"] in (0, 2))              # keep only entail / contradict
                )

                # turn 0 / 2 into binary 1 / 0
                labels = [0 if lbl == 2 else 1 for lbl in ds["label"]]

                new_dataset[lang][split] = [{
                    "sentence1": ds["premise"],       # list[str]
                    "sentence2": ds["hypothesis"],    # list[str]
                    "labels":    labels,              # list[int]
                }]

        self.dataset = new_dataset