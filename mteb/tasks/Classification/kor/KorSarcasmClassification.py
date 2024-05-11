from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
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
            "path": "kor_sarcasm",
            "revision": "8079d24b9f1278c6fbc992921c1271457a1064ff",
        },
        reference="https://github.com/SpellOnYou/korean-sarcasm",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-10-31", "2019-09-28"),  # estimated based on git history
        form=["written"],
        domains=["Social"],
        task_subtypes=["Topic classification"],
        license="MIT",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
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
        n_samples={"train": 9000},
        avg_character_length={"train": 49.24},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"tokens": "text"})
        self.dataset = self.dataset["train"].train_test_split(
            test_size=0.5, seed=self.seed, stratify_by_column="label"
        )
