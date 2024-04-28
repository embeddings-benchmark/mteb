from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AJGT(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AJGT",
        dataset={
            "path": "ajgt_twitter_ar",
            "revision": "af3f2fa5462ac461b696cb300d66e07ad366057f",
        },
        description="Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["ara-Arab"],
        main_score="accuracy",
        date=("2021-01-01", "2022-01-25"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="AFL",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=["ara-arab-MSA", "ara-arab-JO"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{alomari2017arabic,
  title={Arabic tweets sentimental analysis using machine learning},
  author={Alomari, Khaled Mohammad and ElSherif, Hatem M and Shaalan, Khaled},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  pages={602--610},
  year={2017},
  organization={Springer}
}
""",
        n_samples={"train": 1800},
        avg_character_length={"train": 46.81},
    )
