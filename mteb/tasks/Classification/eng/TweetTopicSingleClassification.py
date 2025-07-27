from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TweetTopicSingleClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetTopicSingleClassification",
        description="""Topic classification dataset on Twitter with 6 labels. Each instance of
        TweetTopic comes with a timestamp which distributes from September 2019 to August 2021.
        Tweets were preprocessed before the annotation to normalize some artifacts, converting
        URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified
        usernames, we replace its display name (or account name) with symbols {@}.
        """,
        dataset={
            "path": "cardiffnlp/tweet_topic_single",
            "revision": "87b7a0d1c402dbb481db649569c556d9aa27ac05",
            "trust_remote_code": True,
        },
        reference="https://arxiv.org/abs/2209.09824",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test_2021"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-09-01", "2021-08-31"),
        domains=["Social", "News", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dimosthenis-etal-2022-twitter,
  address = {Gyeongju, Republic of Korea},
  author = {Antypas, Dimosthenis  and
Ushio, Asahi  and
Camacho-Collados, Jose  and
Neves, Leonardo  and
Silva, Vitor  and
Barbieri, Francesco},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  month = oct,
  publisher = {International Committee on Computational Linguistics},
  title = {{T}witter {T}opic {C}lassification},
  year = {2022},
}
""",
    )

    def dataset_transform(self):
        self.dataset["train"] = self.dataset["train_2021"]
