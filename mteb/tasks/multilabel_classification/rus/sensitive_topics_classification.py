from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class SensitiveTopicsClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="SensitiveTopicsClassification",
        dataset={
            "path": "ai-forever/sensitive-topics-classification",
            "revision": "416b34a802308eac30e4192afc0ff99bb8dcc7f2",
        },
        description="Multilabel classification of sentences across 18 sensitive topics.",
        reference="https://aclanthology.org/2021.bsnlp-1.4",
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2006-01-01", "2021-04-01"),
        domains=["Web", "Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{babakov-etal-2021-detecting,
  address = {Kiyv, Ukraine},
  author = {Babakov, Nikolay  and
Logacheva, Varvara  and
Kozlova, Olga  and
Semenov, Nikita  and
Panchenko, Alexander},
  booktitle = {Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing},
  editor = {Babych, Bogdan  and
Kanishcheva, Olga  and
Nakov, Preslav  and
Piskorski, Jakub  and
Pivovarova, Lidia  and
Starko, Vasyl  and
Steinberger, Josef  and
Yangarber, Roman  and
Marci{\'n}czuk, Micha{\l}  and
Pollak, Senja  and
P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
Robnik-{\v{S}}ikonja, Marko},
  month = apr,
  pages = {26--36},
  publisher = {Association for Computational Linguistics},
  title = {Detecting Inappropriate Messages on Sensitive Topics that Could Harm a Company{'}s Reputation},
  url = {https://aclanthology.org/2021.bsnlp-1.4},
  year = {2021},
}
""",
        prompt="Given a sentence as query, find sensitive topics",
        superseded_by="SensitiveTopicsClassification.v2",
    )
    # Published results score the whole evaluation split, so the cap stays off to keep them comparable.
    max_eval_samples: int | None = None


class SensitiveTopicsClassificationV2(SensitiveTopicsClassification):
    metadata = SensitiveTopicsClassification.metadata.model_copy(
        update={
            "name": "SensitiveTopicsClassification.v2",
            "description": SensitiveTopicsClassification.metadata.description
            + " This version scores at most 2000 rows of each evaluation split, sampled with iterative stratification over the labels.",
            "superseded_by": None,
            "adapted_from": ["SensitiveTopicsClassification"],
        },
        deep=True,
    )
    max_eval_samples = 2000
