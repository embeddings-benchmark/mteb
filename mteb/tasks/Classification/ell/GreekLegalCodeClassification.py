from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GreekLegalCodeClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GreekLegalCodeClassification",
        description="Greek Legal Code Dataset for Classification.",
        reference="https://arxiv.org/abs/2109.15298",
        dataset={
            "path": "AI-team-UoA/greek_legal_code",
            "revision": "de0fdb34424f07d1ac6f0ede23ee0ed44bd9f5d1",
        },
        type="Classification",
        category="s2s",
        date=("2021-01-01", "2021-01-01"),
        eval_splits=["validation", "test"],
        eval_langs=["ell-Grek"],
        main_score="accuracy",
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{papaloukas-etal-2021-glc,
    title = "Multi-granular Legal Topic Classification on Greek Legislation",
    author = "Papaloukas, Christos and Chalkidis, Ilias and Athinaios, Konstantinos and Pantazi, Despina-Athanasia and Koubarakis, Manolis",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2109.15298",
    doi = "10.48550/arXiv.2109.15298",
    pages = "63--75"
}
""",
        n_samples={"validation": 9511, "test": 9516},
        avg_character_length={"validation": 4046.8, "test": 4200.8},
    )
