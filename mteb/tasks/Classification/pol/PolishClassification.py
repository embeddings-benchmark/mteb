from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class CbdClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CBD",
        description="Polish Tweets annotated for cyberbullying detection.",
        reference="http://2019.poleval.pl/files/poleval2019.pdf",
        dataset={
            "path": "PL-MTEB/cbd",
            "revision": "36ddb419bcffe6a5374c3891957912892916f28d",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 1000},
        avg_character_length={"test": 93.2},
    )


class PolEmo2InClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-IN",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        "school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.",
        reference="https://aclanthology.org/K19-1092.pdf",
        dataset={
            "path": "PL-MTEB/polemo2_in",
            "revision": "d90724373c70959f17d2331ad51fb60c71176b03",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )


class PolEmo2OutClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-OUT",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        "school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and "
        "school) reviews using models train on reviews from medicine and hotels domains.",
        reference="https://aclanthology.org/K19-1092.pdf",
        dataset={
            "path": "PL-MTEB/polemo2_out",
            "revision": "6a21ab8716e255ab1867265f8b396105e8aa63d4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 722},
        avg_character_length={"test": 756.2},
    )


class AllegroReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AllegroReviews",
        description="A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.",
        reference="https://aclanthology.org/2020.acl-main.111.pdf",
        dataset={
            "path": "PL-MTEB/allegro-reviews",
            "revision": "b89853e6de927b0e3bfa8ecc0e56fe4e02ceafc6",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 1006},
        avg_character_length={"test": 477.2},
    )


class PacClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PAC",
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2211.13112.pdf",
        dataset={
            "path": "laugustyniak/abusive-clauses-pl",
            "revision": "fc69d1c153a8ccdcf1eef52f4e2a27f88782f543",
        },
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 3453},
        avg_character_length={"test": 185.3},
    )
