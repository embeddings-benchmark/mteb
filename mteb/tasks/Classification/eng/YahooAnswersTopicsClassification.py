from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class YahooAnswersTopicsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YahooAnswersTopicsClassification",
        description="Dataset composed of questions and answers from Yahoo Answers, categorized into topics.",
        reference="https://huggingface.co/datasets/yahoo_answers_topics",
        dataset={
            "path": "community-datasets/yahoo_answers_topics",
            "revision": "78fccffa043240c80e17a6b1da724f5a1057e8e5",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        domains=["Web", "Written"],
        task_subtypes=["Topic classification"],
        license="Not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
       @inproceedings{NIPS2015_250cf8b5,
        author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
        pages = {},
        publisher = {Curran Associates, Inc.},
        title = {Character-level Convolutional Networks for Text Classification},
        url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
        volume = {28},
        year = {2015}
        }""",
        descriptive_stats={
            "n_samples": {"test": 60000},
            "avg_character_length": {"test": 346.35},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.remove_columns(
            ["id", "question_title", "question_content"]
        )

        self.dataset = self.dataset.rename_columns(
            {"topic": "label", "best_answer": "text"}
        )

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
