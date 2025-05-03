from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WRIMEClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WRIMEClassification",
        dataset={
            "path": "shunk031/wrime",
            "revision": "3fb7212c389d7818b8e6179e2cdac762f2e081d9",
            "name": "ver2",
            "trust_remote_code": True,
        },
        description="A dataset of Japanese social network rated for sentiment",
        reference="https://aclanthology.org/2021.naacl-main.169/",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="accuracy",
        date=("2011-06-01", "2020-05-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="https://huggingface.co/datasets/shunk031/wrime#licensing-information",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kajiwara-etal-2021-wrime,
  abstract = {We annotate 17,000 SNS posts with both the writer{'}s subjective emotional intensity and the reader{'}s objective one to construct a Japanese emotion analysis dataset. In this study, we explore the difference between the emotional intensity of the writer and that of the readers with this dataset. We found that the reader cannot fully detect the emotions of the writer, especially anger and trust. In addition, experimental results in estimating the emotional intensity show that it is more difficult to estimate the writer{'}s subjective labels than the readers{'}. The large gap between the subjective and objective emotions imply the complexity of the mapping from a post to the subjective emotion intensities, which also leads to a lower performance with machine learning models.},
  address = {Online},
  author = {Kajiwara, Tomoyuki  and
Chu, Chenhui  and
Takemura, Noriko  and
Nakashima, Yuta  and
Nagahara, Hajime},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.169},
  editor = {Toutanova, Kristina  and
Rumshisky, Anna  and
Zettlemoyer, Luke  and
Hakkani-Tur, Dilek  and
Beltagy, Iz  and
Bethard, Steven  and
Cotterell, Ryan  and
Chakraborty, Tanmoy  and
Zhou, Yichao},
  month = jun,
  pages = {2095--2104},
  publisher = {Association for Computational Linguistics},
  title = {{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations},
  url = {https://aclanthology.org/2021.naacl-main.169},
  year = {2021},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.flatten().select_columns(
            ["sentence", "avg_readers.sentiment"]
        )
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text", "avg_readers.sentiment": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
