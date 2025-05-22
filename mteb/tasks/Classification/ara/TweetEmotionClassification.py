from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TweetEmotionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetEmotionClassification",
        dataset={
            "path": "mteb/TweetEmotionClassification",
            "revision": "0d803980e91953cc67c21429f74b301b7b1b3f08",
        },
        description="A dataset of 10,000 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2014-01-01", "2016-08-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=["ara-arab-EG", "ara-arab-LB", "ara-arab-JO", "ara-arab-SA"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{al2018emotional,
  author = {Al-Khatib, Amr and El-Beltagy, Samhaa R},
  booktitle = {Computational Linguistics and Intelligent Text Processing: 18th International Conference, CICLing 2017, Budapest, Hungary, April 17--23, 2017, Revised Selected Papers, Part II 18},
  organization = {Springer},
  pages = {105--114},
  title = {Emotional tone detection in arabic tweets},
  year = {2018},
}
""",
    )
