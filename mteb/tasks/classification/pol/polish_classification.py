from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-12-31"),  # best guess: based on publication date
        domains=["Written", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="bsd-3-clause",
        annotations_creators="human-annotated",  # guess
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@proceedings{ogr:kob:19:poleval,
  address = {Warsaw, Poland},
  editor = {Maciej Ogrodniczuk and Łukasz Kobyliński},
  isbn = {978-83-63159-28-3},
  publisher = {Institute of Computer Science, Polish Academy of Sciences},
  title = {{Proceedings of the PolEval 2019 Workshop}},
  url = {http://2019.poleval.pl/files/poleval2019.pdf},
  year = {2019},
}
""",
        superseded_by="CBD.v2",
    )


class CbdClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CBD.v2",
        description="Polish Tweets annotated for cyberbullying detection. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="http://2019.poleval.pl/files/poleval2019.pdf",
        dataset={
            "path": "mteb/cbd",
            "revision": "d962699e284a173179a05052b49d0a9001a25bc0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-12-31"),  # best guess: based on publication date
        domains=["Written", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="bsd-3-clause",
        annotations_creators="human-annotated",  # guess
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@proceedings{ogr:kob:19:poleval,
  address = {Warsaw, Poland},
  editor = {Maciej Ogrodniczuk and Łukasz Kobyliński},
  isbn = {978-83-63159-28-3},
  publisher = {Institute of Computer Science, Polish Academy of Sciences},
  title = {{Proceedings of the PolEval 2019 Workshop}},
  url = {http://2019.poleval.pl/files/poleval2019.pdf},
  year = {2019},
}
""",
        adapted_from=["CbdClassification"],
    )


class PolEmo2InClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-IN",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        + "school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.",
        reference="https://aclanthology.org/K19-1092.pdf",
        dataset={
            "path": "PL-MTEB/polemo2_in",
            "revision": "d90724373c70959f17d2331ad51fb60c71176b03",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2004-01-01", "2019-05-30"),  # based on plot in paper
        domains=["Written", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kocon-etal-2019-multi,
  address = {Hong Kong, China},
  author = {Koco{\'n}, Jan  and
Mi{\l}kowski, Piotr  and
Za{\'s}ko-Zieli{\'n}ska, Monika},
  booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
  doi = {10.18653/v1/K19-1092},
  month = nov,
  pages = {980--991},
  publisher = {Association for Computational Linguistics},
  title = {Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews},
  url = {https://aclanthology.org/K19-1092},
  year = {2019},
}
""",
        superseded_by="PolEmo2.0-IN.v2",
    )


class PolEmo2InClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-IN.v2",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        + "school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.",
        reference="https://aclanthology.org/K19-1092.pdf",
        dataset={
            "path": "mteb/pol_emo2_in",
            "revision": "15f86f0432cd7c91437cf7c673993527e2f53fd8",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2004-01-01", "2019-05-30"),  # based on plot in paper
        domains=["Written", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kocon-etal-2019-multi,
  address = {Hong Kong, China},
  author = {Koco{\'n}, Jan  and
Mi{\l}kowski, Piotr  and
Za{\'s}ko-Zieli{\'n}ska, Monika},
  booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
  doi = {10.18653/v1/K19-1092},
  month = nov,
  pages = {980--991},
  publisher = {Association for Computational Linguistics},
  title = {Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews},
  url = {https://aclanthology.org/K19-1092},
  year = {2019},
}
""",
        adapted_from=["PolEmo2InClassification"],
    )


class PolEmo2OutClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-OUT",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        + "school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and "
        + "school) reviews using models train on reviews from medicine and hotels domains.",
        reference="https://aclanthology.org/K19-1092.pdf",
        dataset={
            "path": "PL-MTEB/polemo2_out",
            "revision": "6a21ab8716e255ab1867265f8b396105e8aa63d4",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2004-01-01", "2019-05-30"),  # based on plot in paper
        domains=["Written", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        superseded_by="PolEmo2.0-OUT.v2",
    )


class PolEmo2OutClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PolEmo2.0-OUT.v2",
        description="A collection of Polish online reviews from four domains: medicine, hotels, products and "
        + "school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and "
        + "school) reviews using models train on reviews from medicine and hotels domains.",
        reference="https://aclanthology.org/K19-1092.pdf",
        dataset={
            "path": "mteb/pol_emo2_out",
            "revision": "f7f3752b56dcbc4c84077274dfa687efa38476fb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2004-01-01", "2019-05-30"),  # based on plot in paper
        domains=["Written", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        adapted_from=["PolEmo2OutClassification"],
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
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=(
            "2020-06-22",
            "2020-07-07",
        ),  # best guess: based on commit dates in https://github.com/allegro/klejbenchmark-baselines
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{rybak-etal-2020-klej,
  address = {Online},
  author = {Rybak, Piotr  and
Mroczkowski, Robert  and
Tracz, Janusz  and
Gawlik, Ireneusz},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2020.acl-main.111},
  editor = {Jurafsky, Dan  and
Chai, Joyce  and
Schluter, Natalie  and
Tetreault, Joel},
  month = jul,
  pages = {1191--1201},
  publisher = {Association for Computational Linguistics},
  title = {{KLEJ}: Comprehensive Benchmark for {P}olish Language Understanding},
  url = {https://aclanthology.org/2020.acl-main.111/},
  year = {2020},
}
""",
        superseded_by="AllegroReviews.v2",
    )


class AllegroReviewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AllegroReviews.v2",
        description="A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2020.acl-main.111.pdf",
        dataset={
            "path": "mteb/allegro_reviews",
            "revision": "5233456d195235bf93f45b8ef54d72f72957dbf1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=(
            "2020-06-22",
            "2020-07-07",
        ),  # best guess: based on commit dates in https://github.com/allegro/klejbenchmark-baselines
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{rybak-etal-2020-klej,
  address = {Online},
  author = {Rybak, Piotr  and
Mroczkowski, Robert  and
Tracz, Janusz  and
Gawlik, Ireneusz},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2020.acl-main.111},
  editor = {Jurafsky, Dan  and
Chai, Joyce  and
Schluter, Natalie  and
Tetreault, Joel},
  month = jul,
  pages = {1191--1201},
  publisher = {Association for Computational Linguistics},
  title = {{KLEJ}: Comprehensive Benchmark for {P}olish Language Understanding},
  url = {https://aclanthology.org/2020.acl-main.111/},
  year = {2020},
}
""",
        adapted_from=["AllegroReviewsClassification"],
    )


class PacClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PAC",
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2211.13112.pdf",
        dataset={
            "path": "mteb/PAC",
            "revision": "d09e1c9f0f04f7f52f5cbfb74fa6c793c4eb84da",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),  # best guess: based on publication date
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation=r"""
@misc{augustyniak2022waydesigningcompilinglepiszcze,
  archiveprefix = {arXiv},
  author = {Łukasz Augustyniak and Kamil Tagowski and Albert Sawczyn and Denis Janiak and Roman Bartusiak and Adrian Szymczak and Marcin Wątroba and Arkadiusz Janz and Piotr Szymański and Mikołaj Morzy and Tomasz Kajdanowicz and Maciej Piasecki},
  eprint = {2211.13112},
  primaryclass = {cs.CL},
  title = {This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish},
  url = {https://arxiv.org/abs/2211.13112},
  year = {2022},
}
""",
        superseded_by="PAC.v2",
    )


class PacClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PAC.v2",
        description="Polish Paraphrase Corpus This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/pdf/2211.13112.pdf",
        dataset={
            "path": "mteb/pac",
            "revision": "53c98e6a9173c550f1b60f0da9152e67e9618897",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),  # best guess: based on publication date
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation=r"""
@misc{augustyniak2022waydesigningcompilinglepiszcze,
  archiveprefix = {arXiv},
  author = {Łukasz Augustyniak and Kamil Tagowski and Albert Sawczyn and Denis Janiak and Roman Bartusiak and Adrian Szymczak and Marcin Wątroba and Arkadiusz Janz and Piotr Szymański and Mikołaj Morzy and Tomasz Kajdanowicz and Maciej Piasecki},
  eprint = {2211.13112},
  primaryclass = {cs.CL},
  title = {This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish},
  url = {https://arxiv.org/abs/2211.13112},
  year = {2022},
}
""",
        adapted_from=["PacClassification"],
    )
