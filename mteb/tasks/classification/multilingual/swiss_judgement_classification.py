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
@inproceedings{niklaus2022empirical,
  address = {Online only},
  author = {Niklaus, Joel and St{\"u}rmer, Matthias and Chalkidis, Ilias},
  booktitle = {Proceedings of the 2nd Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2022.aacl-main.3},
  editor = {He, Yulan and Ji, Heng and Li, Sujian and Liu, Yang and Chang, Chua-Hui},
  month = {nov},
  pages = {32--46},
  publisher = {Association for Computational Linguistics},
  title = {An Empirical Study on Cross-{X} Transfer for Legal Judgment Prediction},
  url = {https://aclanthology.org/2022.aacl-main.3/},
  year = {2022},
}
""",
    )
