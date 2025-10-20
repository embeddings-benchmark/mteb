from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HateSpeechPortugueseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HateSpeechPortugueseClassification",
        description="HateSpeechPortugueseClassification is a dataset of Portuguese tweets categorized with their sentiment (2 classes).",
        reference="https://aclanthology.org/W19-3510",
        dataset={
            "path": "mteb/HateSpeechPortugueseClassification",
            "revision": "46a9ecdbad49534ea2bd771edde6423e15a2364d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2017-03-08", "2017-03-09"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{fortuna-etal-2019-hierarchically,
  address = {Florence, Italy},
  author = {Fortuna, Paula  and
Rocha da Silva, Jo{\~a}o  and
Soler-Company, Juan  and
Wanner, Leo  and
Nunes, S{\'e}rgio},
  booktitle = {Proceedings of the Third Workshop on Abusive Language Online},
  doi = {10.18653/v1/W19-3510},
  editor = {Roberts, Sarah T.  and
Tetreault, Joel  and
Prabhakaran, Vinodkumar  and
Waseem, Zeerak},
  month = aug,
  pages = {94--104},
  publisher = {Association for Computational Linguistics},
  title = {A Hierarchically-Labeled {P}ortuguese Hate Speech Dataset},
  url = {https://aclanthology.org/W19-3510},
  year = {2019},
}
""",
    )
