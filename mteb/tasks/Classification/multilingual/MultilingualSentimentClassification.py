from __future__ import annotations

from mteb.abstasks import AbsTaskClassification, MultilingualTask, TaskMetadata

_LANGUAGES = {
    "urd": ["urd-Arab"],
    "vie": ["vie-Latn"],
    "dza": ["dza-Arab"],
    "tha": ["tha-Thai"],
    "tur": ["tur-Latn"],
    "slk": ["slk-Latn"],
    "nor": ["nor-Latn"],
    "spa": ["spa-Latn"],
    "rus": ["rus-Cyrl"],
    "mlt": ["mlt-Latn"],
    "kor": ["kor-Hang"],
    "ind": ["ind-Latn"],
    "heb": ["heb-Latn"],
    "jpn": ["jpn-Jpan"],
    "ell": ["ell-Latn"],
    "deu": ["deu-Latn"],
    "eng": ["eng-Latn"],
    "fin": ["fin-Latn"],
    "hrv": ["hrv-Latn"],
    "zho": ["zho-Hans"],
    "cmn": ["cmn-Hans"],
    "bul": ["bul-Cyrl"],
    "eus": ["eus-Latn"],
    "uig": ["uig-Hans"],
    "bam": ["bam-Latn"],
    "pol": ["pol-Latn"],
    # The train set for "cym" language is created from the test set
    "cym": ["cym-Latn"],
    # "hin": ["hin-Deva"], # Do not handle this subset since it does not contain a test set required by the evaluation
    "ara": ["ara-Arab"],
    "fas": ["fas-Arab"],
}


class MultilingualSentimentClassification(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="MultilingualSentimentClassification",
        dataset={
            "path": "mteb/multilingual-sentiment-classification",
            "revision": "67787ff1c3531563c137e77d0d399f21d9e9065f",
        },
        description="""Sentiment classification dataset with binary
                       (positive vs negative sentiment) labels. Includes 30 languages and dialects.
                     """,
        reference="https://huggingface.co/datasets/mteb/multilingual-sentiment-classification",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=["ar-dz"],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{mollanorozy-etal-2023-cross,
            title = "Cross-lingual Transfer Learning with \{P\}ersian",
            author = "Mollanorozy, Sepideh  and
            Tanti, Marc  and
            Nissim, Malvina",
            editor = "Beinborn, Lisa  and
            Goswami, Koustava  and
            Murado{\\u{g}}lu, Saliha  and
            Sorokin, Alexey  and
            Kumar, Ritesh  and
            Shcherbakov, Andreas  and
            Ponti, Edoardo M.  and
            Cotterell, Ryan  and
            Vylomova, Ekaterina",
            booktitle = "Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
            month = may,
            year = "2023",
            address = "Dubrovnik, Croatia",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2023.sigtyp-1.9",
            doi = "10.18653/v1/2023.sigtyp-1.9",
            pages = "89--95",
        }
        """,
        n_samples={"test": 7000},
        avg_character_length={"test": 56},
    )

    def dataset_transform(self):
        # create a train set from the test set for Welsh language (cym)
        lang = "cym"
        _dataset = self.dataset[lang]
        if lang in self.dataset.keys():
            _dataset = _dataset.class_encode_column("label")
            _dataset = _dataset["test"].train_test_split(
                test_size=0.3, seed=self.seed, stratify_by_column="label"
            )
            _dataset = self.stratified_subsampling(
                dataset_dict=_dataset, seed=self.seed, splits=["test"]
            )
        self.dataset[lang] = _dataset
