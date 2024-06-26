from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
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
        eval_splits=["test_2021"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-09-01", "2021-08-31"),
        form=["written"],
        domains=["Social", "News"],
        task_subtypes=["Topic classification"],
        license="Other",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{dimosthenis-etal-2022-twitter,
            title = "{T}witter {T}opic {C}lassification",
            author = "Antypas, Dimosthenis  and
            Ushio, Asahi  and
            Camacho-Collados, Jose  and
            Neves, Leonardo  and
            Silva, Vitor  and
            Barbieri, Francesco",
            booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
            month = oct,
            year = "2022",
            address = "Gyeongju, Republic of Korea",
            publisher = "International Committee on Computational Linguistics"
        }
        """,
        n_samples={"test_2021": 1693},
        avg_character_length={"test_2021": 167.66},
    )

    def dataset_transform(self):
        self.dataset["train"] = self.dataset["train_2021"]
