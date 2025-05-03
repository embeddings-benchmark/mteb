from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
  abstract = {Not all topics are equally {``}flammable{''} in terms of toxicity: a calm discussion of turtles or fishing less often fuels inappropriate toxic dialogues than a discussion of politics or sexual minorities. We define a set of sensitive topics that can yield inappropriate and toxic messages and describe the methodology of collecting and labelling a dataset for appropriateness. While toxicity in user-generated data is well-studied, we aim at defining a more fine-grained notion of inappropriateness. The core of inappropriateness is that it can harm the reputation of a speaker. This is different from toxicity in two respects: (i) inappropriateness is topic-related, and (ii) inappropriate message is not toxic but still unacceptable. We collect and release two datasets for Russian: a topic-labelled dataset and an appropriateness-labelled dataset. We also release pre-trained classification models trained on this data.},
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
    )
