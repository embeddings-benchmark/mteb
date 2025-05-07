from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AJGT(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AJGT",
        dataset={
            "path": "komari6/ajgt_twitter_ar",
            "revision": "af3f2fa5462ac461b696cb300d66e07ad366057f",
        },
        description="Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2021-01-01", "2022-01-25"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="afl-3.0",
        annotations_creators="human-annotated",
        dialect=["ara-arab-MSA", "ara-arab-JO"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{alomari2017arabic,
  author = {Alomari, Khaled Mohammad and ElSherif, Hatem M and Shaalan, Khaled},
  booktitle = {International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  organization = {Springer},
  pages = {602--610},
  title = {Arabic tweets sentimental analysis using machine learning},
  year = {2017},
}
""",
    )
