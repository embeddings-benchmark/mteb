from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "afr": ["afr-Latn"],
    "eng": ["eng-Latn"],
    "nbl": ["nbl-Latn"],
    "nso": ["nso-Latn"],
    "sot": ["sot-Latn"],
    "ssw": ["ssw-Latn"],
    "tsn": ["tsn-Latn"],
    "tso": ["tso-Latn"],
    "ven": ["ven-Latn"],
    "xho": ["xho-Latn"],
    "zul": ["zul-Latn"],
}


class SouthAfricanLangClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SouthAfricanLangClassification",
        dataset={
            "path": "mlexplorer008/south_african_language_identification",
            "revision": "5ccda92ffd7e74fa91fed595a1cbcff1bb68ec2d",
        },
        description="A language identification test set for 11 South African Languages.",
        reference="https://www.kaggle.com/competitions/south-african-language-identification/",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2010-01-01", "2023-01-01"),
        domains=["Web", "Non-fiction", "Written"],
        task_subtypes=["Language identification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{south-african-language-identification,
    author = {ExploreAI Academy, Joanne M},
    title = {South African Language Identification},
    publisher = {Kaggle},
    year = {2022},
    url = {https://kaggle.com/competitions/south-african-language-identification}
}""",
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {" text": "text", "lang_id": "label"}
        )
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
