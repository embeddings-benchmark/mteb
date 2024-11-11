from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
}


class RTE3(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RTE3",
        dataset={
            "path": "maximoss/rte3-multi",
            "revision": "d94f96ca5a6798e20f5a77e566f7a288dc6138d7",
        },
        description="Recognising Textual Entailment Challenge (RTE-3) aim to provide the NLP community with a benchmark to test progress in recognizing textual entailment",
        reference="https://aclanthology.org/W07-1401/",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="max_ap",
        date=("2023-03-25", "2024-04-15"),
        domains=["News", "Web", "Encyclopaedic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{giampiccolo-etal-2007-third,
            title = "The Third {PASCAL} Recognizing Textual Entailment Challenge",
            author = "Giampiccolo, Danilo  and
            Magnini, Bernardo  and
            Dagan, Ido  and
            Dolan, Bill",
            booktitle = "Proceedings of the {ACL}-{PASCAL} Workshop on Textual Entailment and Paraphrasing",
            month = jun,
            year = "2007",
            address = "Prague",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/W07-1401",
            pages = "1--9",
        }
        """,
        # sum of 4 languages after neutral filtering
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(
            self.metadata.dataset["path"], revision=self.metadata.dataset["revision"]
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            for split in self.metadata.eval_splits:
                # keep target language
                hf_dataset = self.dataset[split].filter(lambda x: x["language"] == lang)
                # keep labels 0=entailment and 2=contradiction, and map them as 1 and 0 for binary classification
                hf_dataset = hf_dataset.filter(lambda x: x["label"] in [0, 2])
                hf_dataset = hf_dataset.map(
                    lambda example: {"label": 0 if example["label"] == 2 else 1}
                )
                _dataset[lang][split] = [
                    {
                        "sentence1": hf_dataset["premise"],
                        "sentence2": hf_dataset["hypothesis"],
                        "labels": hf_dataset["label"],
                    }
                ]
        self.dataset = _dataset
