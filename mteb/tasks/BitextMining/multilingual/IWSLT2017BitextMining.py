from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "ar-en": ["ara-Arab", "eng-Latn"],
    "de-en": ["deu-Latn", "eng-Latn"],
    "en-ar": ["eng-Latn", "ara-Arab"],
    "en-de": ["eng-Latn", "deu-Latn"],
    "en-fr": ["eng-Latn", "fra-Latn"],
    "en-it": ["eng-Latn", "ita-Latn"],
    "en-ja": ["eng-Latn", "jpn-Jpan"],
    "en-ko": ["eng-Latn", "kor-Hang"],
    "en-nl": ["eng-Latn", "nld-Latn"],
    "en-ro": ["eng-Latn", "ron-Latn"],
    "en-zh": ["eng-Latn", "cmn-Hans"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "it-en": ["ita-Latn", "eng-Latn"],
    "it-nl": ["ita-Latn", "nld-Latn"],
    "it-ro": ["ita-Latn", "ron-Latn"],
    "ja-en": ["jpn-Jpan", "eng-Latn"],
    "ko-en": ["kor-Hang", "eng-Latn"],
    "nl-en": ["nld-Latn", "eng-Latn"],
    "nl-it": ["nld-Latn", "ita-Latn"],
    "nl-ro": ["nld-Latn", "ron-Latn"],
    "ro-en": ["ron-Latn", "eng-Latn"],
    "ro-it": ["ron-Latn", "ita-Latn"],
    "ro-nl": ["ron-Latn", "nld-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
}

_SPLITS = ["validation"]


class IWSLT2017BitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="IWSLT2017BitextMining",
        dataset={
            "path": "IWSLT/iwslt2017",
            "revision": "c18a4f81a47ae6fa079fe9d32db288ddde38451d",
            "trust_remote_code": True,
        },
        description="The IWSLT 2017 Multilingual Task addresses text translation, including zero-shot translation, with a single MT system across all directions including English, German, Dutch, Italian and Romanian.",
        reference="https://aclanthology.org/2017.iwslt-1.1/",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2007-01-01", "2017-12-14"),  # rough estimate
        domains=["Non-fiction", "Fiction", "Written"],
        task_subtypes=[],
        license="cc-by-nc-nd-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{cettolo-etal-2017-overview,
    title = "Overview of the {IWSLT} 2017 Evaluation Campaign",
    author = {Cettolo, Mauro  and
      Federico, Marcello  and
      Bentivogli, Luisa  and
      Niehues, Jan  and
      St{\"u}ker, Sebastian  and
      Sudoh, Katsuhito  and
      Yoshino, Koichiro  and
      Federmann, Christian},
    editor = "Sakti, Sakriani  and
      Utiyama, Masao",
    booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",
    month = dec # " 14-15",
    year = "2017",
    address = "Tokyo, Japan",
    publisher = "International Workshop on Spoken Language Translation",
    url = "https://aclanthology.org/2017.iwslt-1.1",
    pages = "2--14",
}
""",
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=f"iwslt2017-{lang}",
                **self.metadata_dict["dataset"],
            )

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        def create_columns(row, lang):
            l1, l2 = lang.split("-")
            row["sentence1"] = row["translation"][l1]
            row["sentence2"] = row["translation"][l2]
            return row

        # Convert to standard format
        dataset = {}
        for lang in self.hf_subsets:
            dataset[lang] = {}
            for split in _SPLITS:
                dataset[lang][split] = self.dataset[lang][split].map(
                    lambda x: create_columns(x, lang=lang)
                )
        self.dataset = dataset
