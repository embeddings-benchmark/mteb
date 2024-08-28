from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class KorHateSpeechMLClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="KorHateSpeechMLClassification",
        description="""
        The Korean Multi-label Hate Speech Dataset, K-MHaS, consists of 109,692 utterances from Korean online news comments,
        labelled with 8 fine-grained hate speech classes (labels: Politics, Origin, Physical, Age, Gender, Religion, Race, Profanity)
        or Not Hate Speech class. Each utterance provides from a single to four labels that can handles Korean language patterns effectively.
        For more details, please refer to the paper about K-MHaS, published at COLING 2022.
        This dataset is based on the Korean online news comments available on Kaggle and Github.
        The unlabeled raw data was collected between January 2018 and June 2020.
        The language producers are users who left the comments on the Korean online news platform between 2018 and 2020. 
        """,
        dataset={
            "path": "jeanlee/kmhas_korean_hate_speech",
            "revision": "c657d15baf277c48d467f0625f7d33c50d4352ef",
            "trust_remote_code": True,
        },
        reference="https://paperswithcode.com/dataset/korean-multi-label-hate-speech-dataset",
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-01-01", "2020-06-30"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{lee-etal-2022-k,
            title = "K-{MH}a{S}: A Multi-label Hate Speech Detection Dataset in {K}orean Online News Comment",
            author = "Lee, Jean  and
            Lim, Taejun  and
            Lee, Heejun  and
            Jo, Bogeun  and
            Kim, Yangsok  and
            Yoon, Heegeun  and
            Han, Soyeon Caren",
            booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
            month = oct,
            year = "2022",
            address = "Gyeongju, Republic of Korea",
            publisher = "International Committee on Computational Linguistics",
            url = "https://aclanthology.org/2022.coling-1.311",
            pages = "3530--3538",
        }""",
        descriptive_stats={
            "n_samples": {"train": 8192, "test": 2048},
            "avg_character_length": {"train": 33.67, "test": 34.67},
        },
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], n_samples=8192
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
