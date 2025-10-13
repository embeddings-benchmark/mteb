from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class OpusSlovakEnglishBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="OpusSlovakEnglishBitextMining",
        dataset={
            "path": "Helsinki-NLP/opus-100",
            "revision": "805090d116cbfbb8a4808d8b091fe1f99b7a9679",
            "name": "en-sk",
        },
        description="Slovak-English parallel sentences from OPUS-100, a multilingual dataset with 100 languages for evaluating massively multilingual neural machine translation and zero-shot translation performance.",
        reference="https://aclanthology.org/2020.acl-main.148/",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="f1",
        date=("2000-01-01", "2020-12-31"),  # OPUS collection timeframe
        domains=["Web", "Subtitles", "Fiction", "Non-fiction"],
        task_subtypes=[],
        license="unknown",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{zhang2020improving,
    title = "Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation",
    author = "Zhang, Biao  and
      Williams, Philip  and
      Titov, Ivan  and
      Sennrich, Rico",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.148",
    doi = "10.18653/v1/2020.acl-main.148",
    pages = "1628--1639",
}
""",
        prompt="Retrieve parallel sentences.",
    )

    def dataset_transform(self):
        # Convert from OPUS-100 format to standard bitext format
        # OPUS-100 has structure: {"translation": {"en": "...", "sk": "..."}}
        # We need: {"sentence1": "...", "sentence2": "..."}

        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda x: {
                    "sentence1": x["translation"]["en"],
                    "sentence2": x["translation"]["sk"],
                },
                remove_columns=["translation"],
            )
