from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KorHateClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorHateClassification",
        description="""The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers. 
        """,
        dataset={
            "path": "inmoonlight/kor_hate",
            "revision": "bd1a7370caf712125fac1fda375834ca8ddefaca",
            "trust_remote_code": True,
        },
        reference="https://paperswithcode.com/dataset/korean-hatespeech-dataset",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2018-01-01", "2020-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @misc{moon2020beep,
            title={BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection}, 
            author={Jihyung Moon and Won Ik Cho and Junbum Lee},
            year={2020},
            eprint={2005.12503},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }""",
        descriptive_stats={
            "n_samples": {"train": 2048, "test": 471},
            "avg_character_length": {"train": 38.57, "test": 38.86},
        },
    )

    def dataset_transform(self):
        keep_cols = ["comments", "hate"]
        rename_dict = dict(zip(keep_cols, ["text", "label"]))
        remove_cols = [
            col for col in self.dataset["test"].column_names if col not in keep_cols
        ]
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.dataset.remove_columns(remove_cols)
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
