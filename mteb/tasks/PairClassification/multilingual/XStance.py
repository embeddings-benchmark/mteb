from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class XStance(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XStance",
        dataset={
            "path": "x_stance",
            "revision": "810604b9ad3aafdc6144597fdaa40f21a6f5f3de",
        },
        description="A Multilingual Multi-Target Dataset for Stance Detection in French, German, and Italian.",
        reference="https://github.com/ZurichNLP/xstance",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs={
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "it": ["ita-Latn"],
        },
        main_score="ap",
        date=("2011-01-01", "2020-12-31"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Political classification"],
        license="cc by-nc 4.0",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
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
        n_samples={"test": 2048},
        avg_character_length={"test": 152.41},  # length of`sent1` + `sent2`
    )

    def dataset_transform(self):
        max_n_samples = 2048
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            for split in self.metadata.eval_splits:
                # hf config does not support selection by language, use manual filter instead
                hf_dataset = self.dataset["de"][split].filter(
                    lambda row: row["language"] == lang
                )

                if len(hf_dataset) > max_n_samples:
                    # only de + fr are larger than 2048 samples
                    hf_dataset = hf_dataset.select(range(max_n_samples))

                _dataset[lang][split] = [
                    {
                        "sent1": hf_dataset["question"],
                        "sent2": hf_dataset["comment"],
                        # convert categorical labels into numerical ones
                        "labels": [
                            1 if label == "FAVOR" else 0
                            for label in hf_dataset["label"]
                        ],
                    }
                ]
        self.dataset = _dataset
