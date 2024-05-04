from __future__ import annotations

from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class SwissJudgementClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="SwissJudgementClassification",
        description="Multilingual, diachronic dataset of Swiss Federal Supreme Court cases annotated with the respective binarized judgment outcome (approval/dismissal)",
        reference="https://aclanthology.org/2021.nllp-1.3/",
        dataset={
            "path": "rcds/swiss_judgment_prediction",
            "revision": "29806f87bba4f23d0707d3b6d9ea5432afefbe2f",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "it": ["ita-Latn"],
        },
        main_score="accuracy",
        date=("2020-12-15", "2022-04-08"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[
            "Political classification",
        ],
        license="CC-BY-4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{niklaus2022empirical,
    title={An Empirical Study on Cross-X Transfer for Legal Judgment Prediction},
    author={Joel Niklaus and Matthias Stürmer and Ilias Chalkidis},
    year={2022},
    eprint={2209.12325},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
""",
        n_samples={"test": 17357},
        avg_character_length={"test": 3411.72},
    )

    def dataset_transform(self):
        for lang in self.langs:
            dataset = self.dataset[lang]["test"]
            dataset_dict = {"test": dataset}

            subsampled_dataset_dict = self.stratified_subsampling(
                dataset_dict=dataset_dict,
                seed=42,
                splits=["test"],
                label="label",
                n_samples=min(2048, len(dataset["text"])) - 2,
            )

            self.dataset[lang]["test"] = subsampled_dataset_dict["test"]
