from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SwissJudgementClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwissJudgementClassification",
        description="Multilingual, diachronic dataset of Swiss Federal Supreme Court cases annotated with the respective binarized judgment outcome (approval/dismissal)",
        reference="https://aclanthology.org/2021.nllp-1.3/",
        dataset={
            "path": "mteb/SwissJudgementClassification",
            "revision": "2541705ce837c495e85aa3c44b41a92fa4bd12a8",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "it": ["ita-Latn"],
        },
        main_score="accuracy",
        date=("2020-12-15", "2022-04-08"),
        domains=["Legal", "Written"],
        task_subtypes=[
            "Political classification",
        ],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{niklaus2022empirical,
  archiveprefix = {arXiv},
  author = {Joel Niklaus and Matthias Stürmer and Ilias Chalkidis},
  eprint = {2209.12325},
  primaryclass = {cs.CL},
  title = {An Empirical Study on Cross-X Transfer for Legal Judgment Prediction},
  year = {2022},
}
""",
    )
