from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class XStance(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XStance",
        dataset={
            "path": "ZurichNLP/x_stance",
            "revision": "810604b9ad3aafdc6144597fdaa40f21a6f5f3de",
            "trust_remote_code": True,
        },
        description="A Multilingual Multi-Target Dataset for Stance Detection in French, German, and Italian.",
        reference="https://github.com/ZurichNLP/xstance",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs={
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "it": ["ita-Latn"],
        },
        main_score="max_ap",
        date=("2011-01-01", "2020-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Political classification"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""
            @inproceedings{vamvas2020xstance,
                author    = "Vamvas, Jannis and Sennrich, Rico",
                title     = "{X-Stance}: A Multilingual Multi-Target Dataset for Stance Detection",
                booktitle = "Proceedings of the 5th Swiss Text Analytics Conference (SwissText)  16th Conference on Natural Language Processing (KONVENS)",
                address   = "Zurich, Switzerland",
                year      = "2020",
                month     = "jun",
                url       = "http://ceur-ws.org/Vol-2624/paper9.pdf"
            }
        """,
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {"test": 152.41},
        },  # length of`sent1` + `sent2`
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        max_n_samples = 2048
        self.dataset = {}
        path = self.metadata_dict["dataset"]["path"]
        revision = self.metadata_dict["dataset"]["revision"]
        raw_dataset = load_dataset(path, revision=revision)

        def convert_example(example):
            return {
                "sentence1": example["question"],
                "sentence2": example["comment"],
                "labels": 1 if example["label"] == "FAVOR" else 0,
            }

        for lang in self.metadata.eval_langs:
            self.dataset[lang] = {}
            for split in self.metadata_dict["eval_splits"]:
                # filter by language
                self.dataset[lang][split] = raw_dataset[split].filter(
                    lambda row: row["language"] == lang
                )

                # reduce samples
                if len(self.dataset[lang][split]) > max_n_samples:
                    # only de + fr are larger than 2048 samples
                    self.dataset[lang][split] = self.dataset[lang][split].select(
                        range(max_n_samples)
                    )

                # convert examples
                self.dataset[lang][split] = self.dataset[lang][split].map(
                    convert_example,
                    remove_columns=self.dataset[lang][split].column_names,
                )

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """Transform dataset into sentence-pair format"""
        _dataset = {}

        for lang in self.metadata.eval_langs:
            _dataset[lang] = {}
            for split in self.metadata.eval_splits:
                _dataset[lang][split] = [
                    {
                        "sent1": self.dataset[lang][split]["sent1"],
                        "sent2": self.dataset[lang][split]["sent2"],
                        "labels": self.dataset[lang][split]["labels"],
                    }
                ]
        self.dataset = _dataset
