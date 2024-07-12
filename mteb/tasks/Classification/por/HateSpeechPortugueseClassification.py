from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class HateSpeechPortugueseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HateSpeechPortugueseClassification",
        description="HateSpeechPortugueseClassification is a dataset of Portuguese tweets categorized with their sentiment (2 classes).",
        reference="https://aclanthology.org/W19-3510",
        dataset={
            "path": "hate-speech-portuguese/hate_speech_portuguese",
            "revision": "b0f431acbf8d3865cb7c7b3effb2a9771a618ebc",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2017-03-08", "2017-03-09"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{fortuna-etal-2019-hierarchically,
                title = "A Hierarchically-Labeled {P}ortuguese Hate Speech Dataset",
                author = "Fortuna, Paula  and
                Rocha da Silva, Jo{\~a}o  and
                Soler-Company, Juan  and
                Wanner, Leo  and
                Nunes, S{\'e}rgio",
                editor = "Roberts, Sarah T.  and
                Tetreault, Joel  and
                Prabhakaran, Vinodkumar  and
                Waseem, Zeerak",
                booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
                month = aug,
                year = "2019",
                address = "Florence, Italy",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/W19-3510",
                doi = "10.18653/v1/W19-3510",
                pages = "94--104",
            }
        """,
        descriptive_stats={
            "n_samples": {"train": 2048},
            "avg_character_length": {"train": 101.02},
        },
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
