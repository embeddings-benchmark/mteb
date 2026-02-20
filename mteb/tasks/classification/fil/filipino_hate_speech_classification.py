from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class FilipinoHateSpeechClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FilipinoHateSpeechClassification",
        description="Filipino Twitter dataset for sentiment classification.",
        reference="https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019",
        dataset={
            "path": "mteb/FilipinoHateSpeechClassification",
            "revision": "087a17c0b7f9a78901c88aea00ad2892a319fdac",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-08-01", "2019-08-01"),
        eval_splits=["validation", "test"],
        eval_langs=["fil-Latn"],
        main_score="accuracy",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Cabasag-2019-hate-speech,
  author = {Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
  journal = {Philippine Computing Journal},
  month = {August},
  number = {1},
  title = {Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
  volume = {XIV},
  year = {2019},
}
""",
        superseded_by="FilipinoHateSpeechClassification.v2",
    )


class FilipinoHateSpeechClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FilipinoHateSpeechClassification.v2",
        description="Filipino Twitter dataset for sentiment classification. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019",
        dataset={
            "path": "mteb/filipino_hate_speech",
            "revision": "fdb7536c08d2f10d8b8b618c3ff00b0be286d844",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-08-01", "2019-08-01"),
        eval_splits=["validation", "test"],
        eval_langs=["fil-Latn"],
        main_score="accuracy",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Cabasag-2019-hate-speech,
  author = {Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
  journal = {Philippine Computing Journal},
  month = {August},
  number = {1},
  title = {Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
  volume = {XIV},
  year = {2019},
}
""",
        adapted_from=["FilipinoHateSpeechClassification"],
    )
