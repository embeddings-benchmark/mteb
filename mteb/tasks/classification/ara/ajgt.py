from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
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
        superseded_by="AJGT.v2",
    )


class AJGTV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AJGT.v2",
        dataset={
            "path": "mteb/ajgt",
            "revision": "0a3dea7301ee0c051891f04d32f3e8577a9eae36",
        },
        description="Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets (900 for training and 900 for testing) annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
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
