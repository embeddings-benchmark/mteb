from __future__ import annotations

from mteb.abstasks.AbsTaskAnyClassification import AbsTaskAnyClassification
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class FilipinoHateSpeechClassification(AbsTaskAnyClassification):
    metadata = TaskMetadata.model_construct(
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
    )
