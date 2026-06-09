from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


class OpusSlovakEnglishBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="OpusSlovakEnglishBitextMining",
        dataset={
            "path": "Helsinki-NLP/opus-100",
            "revision": "805090dc28bf78897da9641cdf08b61287580df9",
            "name": "en-sk",
        },
        description="Slovak-English parallel sentences from OPUS-100, a multilingual dataset with 100 languages for evaluating massively multilingual neural machine translation and zero-shot translation performance.",
        reference="https://aclanthology.org/2020.acl-main.148/",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="f1",
        date=("2000-01-01", "2020-12-31"),  # OPUS collection timeframe
        domains=["Web", "Subtitles", "Fiction", "Non-fiction"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{zhang2020improving,
  address = {Online},
  author = {Zhang, Biao  and
Williams, Philip  and
Titov, Ivan  and
Sennrich, Rico},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2020.acl-main.148},
  month = jul,
  pages = {1628--1639},
  publisher = {Association for Computational Linguistics},
  title = {Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation},
  url = {https://aclanthology.org/2020.acl-main.148},
  year = {2020},
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
