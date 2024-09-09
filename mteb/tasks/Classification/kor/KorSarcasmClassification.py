from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KorSarcasmClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorSarcasmClassification",
        description="""
        The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original
        meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These
        tweets were gathered by querying for: irony sarcastic, and
        sarcasm. 
        The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm
        and variants of it were used to return tweets. It was preprocessed by removing the keyword
        hashtag, urls and mentions of the user to preserve anonymity.
        """,
        dataset={
            "path": "SpellOnYou/kor_sarcasm",
            "revision": "8079d24b9f1278c6fbc992921c1271457a1064ff",
            "trust_remote_code": True,
        },
        reference="https://github.com/SpellOnYou/korean-sarcasm",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-10-31", "2019-09-28"),  # estimated based on git history
        domains=["Social", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @misc{kim2019kocasm,
            author = {Kim, Jiwon and Cho, Won Ik},
            title = {Kocasm: Korean Automatic Sarcasm Detection},
            year = {2019},
            publisher = {GitHub},
            journal = {GitHub repository},
            howpublished = {https://github.com/SpellOnYou/korean-sarcasm}
        }
        """,
        descriptive_stats={
            "n_samples": {"train": 2048, "test": 301},
            "avg_character_length": {"train": 48.45, "test": 46.77},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"tokens": "text"})
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
