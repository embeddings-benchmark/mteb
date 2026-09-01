from __future__ import annotations

from typing import ClassVar

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{tonja2026afrimcqa,
  author = {Tonja, Atnafu Lambebo and Anand, Srija and Villa-Cueva, Emilio and Azime, Israel Abebe and Alabi, Jesujoba Oluwadara and Mohamed, Muhidin A. and Yadeta, Debela Desalegn and Abadi, Negasi Haile and Oppong, Abigail and Obiefuna, Nnaemeka Casmir and Abdulmumin, Idris and Etori, Naome A},
  title = {{Afri-MCQA}: Multimodal Cultural Question Answering for {African} Languages},
  year = {2026},
}
"""


class AfriMCQACategoryClassification(AbsTaskClassification):
    input_column_name: ClassVar[tuple[str, str]] = ("image", "audio")
    samples_per_label: int = 8
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="AfriMCQACategoryClassification",
        description=(
            "Classify the cultural category of an Afri-MCQA entry from its photograph "
            "and the question about it spoken by a native speaker, across 16 African "
            "languages. Neither channel is sufficient on its own: the photograph fixes "
            "the subject while the spoken question says what about it is being asked. "
            "The official dev split is used to fit the evaluator and the official test "
            "split is scored. Entries are kept only when both the image and the "
            "recording are present and the entry carries a single category, since "
            "around 13% are tagged with several at once. Construction script: "
            "scripts/data/afri_mcqa_classification/create_data.py."
        ),
        reference="https://arxiv.org/abs/2601.05699",
        dataset={
            "path": "vnahata/AfriMCQA-category-classification",
            "revision": "1581d446105e451a1fc8d2c62fd1f3d54d75fb5e",
        },
        type="ImageClassification",
        category="ia2c",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs={
            "twi": ["twi-Latn"],
            "amh": ["amh-Ethi"],
            "nya": ["nya-Latn"],
            "hau": ["hau-Latn"],
            "ibo": ["ibo-Latn"],
            "kik": ["kik-Latn"],
            "kin": ["kin-Latn"],
            "lin": ["lin-Latn"],
            "lug": ["lug-Latn"],
            # Afri-MCQA's Oromo is the West Central variety, so `gaz` rather than `orm`
            "orm": ["gaz-Latn"],
            "sot": ["sot-Latn"],
            "tsn": ["tsn-Latn"],
            "som": ["som-Latn"],
            "tir": ["tir-Ethi"],
            "yor": ["yor-Latn"],
            "zul": ["zul-Latn"],
        },
        main_score="f1",
        date=("2025-01-01", "2026-01-15"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )
