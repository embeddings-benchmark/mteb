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
        bibtex_citation="""@proceedings{ogr:kob:19:poleval,
  editor    = {Maciej Ogrodniczuk and Łukasz Kobyliński},
  title     = {{Proceedings of the PolEval 2019 Workshop}},
  year      = {2019},
  address   = {Warsaw, Poland},
  publisher = {Institute of Computer Science, Polish Academy of Sciences},
  url       = {http://2019.poleval.pl/files/poleval2019.pdf},
  isbn      = "978-83-63159-28-3"}
}""",
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
        bibtex_citation="""@inproceedings{kocon-etal-2019-multi,
    title = "Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews",
    author = "Koco{\'n}, Jan  and
      Mi{\l}kowski, Piotr  and
      Za{\'s}ko-Zieli{\'n}ska, Monika",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/K19-1092",
    doi = "10.18653/v1/K19-1092",
    pages = "980--991",
    abstract = "In this article we present an extended version of PolEmo {--} a corpus of consumer reviews from 4 domains: medicine, hotels, products and school. Current version (PolEmo 2.0) contains 8,216 reviews having 57,466 sentences. Each text and sentence was manually annotated with sentiment in 2+1 scheme, which gives a total of 197,046 annotations. We obtained a high value of Positive Specific Agreement, which is 0.91 for texts and 0.88 for sentences. PolEmo 2.0 is publicly available under a Creative Commons copyright license. We explored recent deep learning approaches for the recognition of sentiment, such as Bi-directional Long Short-Term Memory (BiLSTM) and Bidirectional Encoder Representations from Transformers (BERT).",
}""",
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
