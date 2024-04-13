from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WRIMEClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WRIMEClassification",
        dataset={
            "path": "shunk031/wrime",
            "revision": "3fb7212c389d7818b8e6179e2cdac762f2e081d9",
            "name": "ver2"
        },
        description="A dataset of Japanese social network rated for sentiment",
        reference="https://aclanthology.org/2021.naacl-main.169/",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="accuracy",
        date=("2011-06-01", "2020-05-31"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="The dataset is available for research purposes only. Redistribution of the dataset is prohibited.",
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@inproceedings{kajiwara-etal-2021-wrime,
    title = "{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations",
    author = "Kajiwara, Tomoyuki  and
      Chu, Chenhui  and
      Takemura, Noriko  and
      Nakashima, Yuta  and
      Nagahara, Hajime",
    editor = "Toutanova, Kristina  and
      Rumshisky, Anna  and
      Zettlemoyer, Luke  and
      Hakkani-Tur, Dilek  and
      Beltagy, Iz  and
      Bethard, Steven  and
      Cotterell, Ryan  and
      Chakraborty, Tanmoy  and
      Zhou, Yichao",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.169",
    doi = "10.18653/v1/2021.naacl-main.169",
    pages = "2095--2104",
    abstract = "We annotate 17,000 SNS posts with both the writer{'}s subjective emotional intensity and the reader{'}s objective one to construct a Japanese emotion analysis dataset. In this study, we explore the difference between the emotional intensity of the writer and that of the readers with this dataset. We found that the reader cannot fully detect the emotions of the writer, especially anger and trust. In addition, experimental results in estimating the emotional intensity show that it is more difficult to estimate the writer{'}s subjective labels than the readers{'}. The large gap between the subjective and objective emotions imply the complexity of the mapping from a post to the subjective emotion intensities, which also leads to a lower performance with machine learning models.",
}""",
        n_samples={"test": 9010}, # TODO
        avg_character_length={"test": 69.9}, # TODO
    )

    def dataset_transform(self):
        
        self.dataset = self.dataset.flatten().select_columns(['sentence', 'avg_readers.sentiment'])
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("avg_readers.sentiment", "label")
        # random downsample to 2048 
        self.dataset['test'] = self.dataset['test'].shuffle(seed=42)
        max_samples = min(2048, len(self.dataset['test']))
        self.dataset['test'] = self.dataset['test'].select(
            range(max_samples)
        )
        