from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class InappropriatenessClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="InappropriatenessClassification",
        dataset={
            "path": "ai-forever/inappropriateness-classification",
            "revision": "601651fdc45ef243751676e62dd7a19f491c0285",
        },
        description="Inappropriateness identification in the form of binary classification",
        reference="https://aclanthology.org/2021.bsnlp-1.4",
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2006-01-01", "2021-04-01"),
        form=["written"],
        domains=["Web", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{babakov-etal-2021-detecting,
        title = "Detecting Inappropriate Messages on Sensitive Topics that Could Harm a Company{'}s Reputation",
        author = "Babakov, Nikolay  and
        Logacheva, Varvara  and
        Kozlova, Olga  and
        Semenov, Nikita  and
        Panchenko, Alexander",
        editor = "Babych, Bogdan  and
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
        Robnik-{\v{S}}ikonja, Marko",
        booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
        month = apr,
        year = "2021",
        address = "Kiyv, Ukraine",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.bsnlp-1.4",
        pages = "26--36",
        abstract = "Not all topics are equally {``}flammable{''} in terms of toxicity: a calm discussion of turtles or fishing less often fuels inappropriate toxic dialogues than a discussion of politics or sexual minorities. We define a set of sensitive topics that can yield inappropriate and toxic messages and describe the methodology of collecting and labelling a dataset for appropriateness. While toxicity in user-generated data is well-studied, we aim at defining a more fine-grained notion of inappropriateness. The core of inappropriateness is that it can harm the reputation of a speaker. This is different from toxicity in two respects: (i) inappropriateness is topic-related, and (ii) inappropriate message is not toxic but still unacceptable. We collect and release two datasets for Russian: a topic-labelled dataset and an appropriateness-labelled dataset. We also release pre-trained classification models trained on this data.",
        }""",
        n_samples={"validation": 4000, "test": 10000},
        avg_character_length={"validation": 96.8, "test": 97.7},
    )
