from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TweetTopicSingleClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetTopicSingleClassification",
        description="Topic classification dataset on Twitter with 6 labels. Each instance of TweetTopic comes with a timestamp which distributes from September 2019 to August 2021. Tweets were preprocessed before the annotation to normalize some artifacts, converting URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified usernames, we replace its display name (or account name) with symbols {@}.",
        dataset={
            "path": "mteb/TweetTopicSingleClassification",
            "revision": "b4280e921a2760ce34d2dd80a9e5dc8bcbf61785",
        },
        reference="https://arxiv.org/abs/2209.09824",
        type="Classification",
        category="t2c",
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
@inproceedings{antypas-etal-2022-twitter,
  address = {Gyeongju, Republic of Korea},
  archiveprefix = {arXiv},
  author = {Antypas, Dimosthenis  and
Ushio, Asahi  and
Camacho-Collados, Jose  and
Silva, Vitor  and
Neves, Leonardo  and
Barbieri, Francesco},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  editor = {Calzolari, Nicoletta  and
Huang, Chu-Ren  and
Kim, Hansaem  and
Pustejovsky, James  and
Wanner, Leo  and
Choi, Key-Sun  and
Ryu, Pum-Mo  and
Chen, Hsin-Hsi  and
Donatelli, Lucia  and
Ji, Heng  and
Kurohashi, Sadao  and
Paggio, Patrizia  and
Xue, Nianwen  and
Kim, Seokhwan  and
Hahm, Younggyun  and
He, Zhong  and
Lee, Tony Kyungil  and
Santus, Enrico  and
Bond, Francis  and
Na, Seung-Hoon},
  eprint = {2209.09824},
  month = oct,
  pages = {3386--3400},
  publisher = {International Committee on Computational Linguistics},
  title = {{T}witter Topic Classification},
  url = {https://aclanthology.org/2022.coling-1.299/},
  year = {2022},
}
""",
        superseded_by="TweetTopicSingleClassification.v2",
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        self.dataset["train"] = self.dataset["train_2021"]


class TweetTopicSingleClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetTopicSingleClassification.v2",
        description="Topic classification dataset on Twitter with 6 labels. Each instance of TweetTopic comes with a timestamp which distributes from September 2019 to August 2021. Tweets were preprocessed before the annotation to normalize some artifacts, converting URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified usernames, we replace its display name (or account name) with symbols {@}. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/tweet_topic_single",
            "revision": "a7904e26081f987da81ad2cc063e09e714e875d0",
        },
        reference="https://arxiv.org/abs/2209.09824",
        type="Classification",
        category="t2c",
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
@inproceedings{antypas-etal-2022-twitter,
  address = {Gyeongju, Republic of Korea},
  archiveprefix = {arXiv},
  author = {Antypas, Dimosthenis  and
Ushio, Asahi  and
Camacho-Collados, Jose  and
Silva, Vitor  and
Neves, Leonardo  and
Barbieri, Francesco},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  editor = {Calzolari, Nicoletta  and
Huang, Chu-Ren  and
Kim, Hansaem  and
Pustejovsky, James  and
Wanner, Leo  and
Choi, Key-Sun  and
Ryu, Pum-Mo  and
Chen, Hsin-Hsi  and
Donatelli, Lucia  and
Ji, Heng  and
Kurohashi, Sadao  and
Paggio, Patrizia  and
Xue, Nianwen  and
Kim, Seokhwan  and
Hahm, Younggyun  and
He, Zhong  and
Lee, Tony Kyungil  and
Santus, Enrico  and
Bond, Francis  and
Na, Seung-Hoon},
  eprint = {2209.09824},
  month = oct,
  pages = {3386--3400},
  publisher = {International Committee on Computational Linguistics},
  title = {{T}witter Topic Classification},
  url = {https://aclanthology.org/2022.coling-1.299/},
  year = {2022},
}
""",
        adapted_from=["TweetTopicSingleClassification"],
    )
