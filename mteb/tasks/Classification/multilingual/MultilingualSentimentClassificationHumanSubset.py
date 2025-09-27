from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "eng": ["eng-Latn"],
    "ara": ["ara-Arab"],
    "nor": ["nor-Latn"],
    "rus": ["rus-Cyrl"],
}


class MultilingualSentimentClassificationHumanSubset(
    AbsTaskClassification, MultilingualTask
):
    fast_loading = True
    metadata = TaskMetadata(
        name="MultilingualSentimentClassificationHumanSubset",
        dataset={
            "path": "mteb/mteb-human-multilingual-sentiment-classification",
            "revision": "fb04d201d89e700f004e5efdb61d9a27e53b728b",
        },
        description="""Human evaluation subset of Sentiment classification dataset with binary
                       (positive vs negative sentiment) labels. Includes 4 languages.
                     """,
        reference="https://huggingface.co/datasets/mteb/multilingual-sentiment-classification",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=["ar-dz"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{mollanorozy-etal-2023-cross,
  address = {Dubrovnik, Croatia},
  author = {Mollanorozy, Sepideh  and
Tanti, Marc  and
Nissim, Malvina},
  booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
  doi = {10.18653/v1/2023.sigtyp-1.9},
  editor = {Beinborn, Lisa  and
Goswami, Koustava  and
Murado{\\u{g}}lu, Saliha  and
Sorokin, Alexey  and
Kumar, Ritesh  and
Shcherbakov, Andreas  and
Ponti, Edoardo M.  and
Cotterell, Ryan  and
Vylomova, Ekaterina},
  month = may,
  pages = {89--95},
  publisher = {Association for Computational Linguistics},
  title = {Cross-lingual Transfer Learning with \{P\}ersian},
  url = {https://aclanthology.org/2023.sigtyp-1.9},
  year = {2023},
}
""",
        adapted_from=["MultilingualSentimentClassification"],
    )

    def load_data(self, **kwargs):
        """Load human test subset + full original training data for each language"""
        # Load human evaluation subset (unified with lang column)
        human_dataset = load_dataset(
            self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
        )

        # Load full original training data (unified with lang column)
        original_dataset = load_dataset("mteb/multilingual-sentiment-classification")

        # Both datasets have unified structure with lang column
        # Split by language to create individual configs
        combined_dataset = {}

        # Filter training data by language
        train_data = original_dataset["train"]
        test_data = human_dataset["test"]

        # Create individual language configs
        for lang in ["eng", "ara", "nor", "rus"]:
            # Filter train data for this language
            train_lang_data = train_data.filter(lambda x: x["lang"] == lang)
            # Filter test data for this language
            test_lang_data = test_data.filter(lambda x: x["lang"] == lang)

            if len(test_lang_data) > 0:  # Only create config if we have test data
                combined_dataset[lang] = DatasetDict(
                    {"train": train_lang_data, "test": test_lang_data}
                )

        self.dataset = combined_dataset
