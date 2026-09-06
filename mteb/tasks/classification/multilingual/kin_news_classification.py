from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class KinNewsClassification(AbsTaskClassification):
    """
    KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi.
    Each sample is a news article from Rwanda and Burundi news websites and newspapers,
    classified into one of 14 possible classes.
    """

    metadata = TaskMetadata(
        name="KinNewsClassification",
        description=(
            "Kinyarwanda and Kirundi news classification datasets (KINNEWS and KIRNEWS, respectively), "
            "which were both collected from Rwanda and Burundi news websites and newspapers, "
            "for low-resource monolingual and cross-lingual multiclass classification tasks. "
            "Each news article can be classified into one of 14 possible classes: politics, sport, "
            "economy, health, entertainment, history, technology, culture, religion, environment, "
            "education, relationship."
        ),
        reference="https://arxiv.org/abs/2010.12174",
        dataset={
            "path": "mteb/KinNewsClassification",
            "revision": "87c7621def3ed9d06e305037416f82eddb9dceac",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "kinnews_cleaned": ["kin-Latn"],  # Kinyarwanda
            "kirnews_cleaned": ["run-Latn"],  # Kirundi (ISO-639-3: run)
        },
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@article{niyongabo2020kinnews,
  author = {Niyongabo, Rubungo Andre and Qu, Hong and Kreutzer, Julia and Huang, Li},
  journal = {arXiv preprint arXiv:2010.12174},
  title = {KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi},
  year = {2020},
}
""",
    )
