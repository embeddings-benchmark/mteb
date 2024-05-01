from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

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
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2010-01-01", "2023-01-01"),
        form=["written"],
        domains=["Web", "Non-fiction"],
        task_subtypes=["Language identification"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 2048},
        avg_character_length={"test": 247.49},
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {" text": "text", "lang_id": "label"}
        )
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
