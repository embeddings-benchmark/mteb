from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxCelebSA(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="VoxCelebSA",
        description="VoxCeleb dataset augmented for Sentiment Analysis task",
        reference="https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
        dataset={
            "path": "DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
            "revision": "554ad4367e98b7c6f4d4d9756dc6bbdf345e042e",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-06-27", "2024-06-28"),
        domains=["Spoken"],
        task_subtypes=["Sentiment Analysis"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@misc{shon2022sluenewbenchmarktasks,
            title={SLUE: New Benchmark Tasks for Spoken Language Understanding Evaluation on Natural Speech}, 
            author={Suwon Shon and Ankita Pasad and Felix Wu and Pablo Brusco and Yoav Artzi and Karen Livescu and Kyu J. Han},
            year={2022},
            eprint={2111.10367},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2111.10367}, 
        }""",
        descriptive_stats={
            "n_samples": {
                "train": 3449
            },  # after removing Disagreement data (before: 3553)
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True

    def dataset_transform(self):
        ## remove disagreement data
        self.dataset = self.dataset.filter(lambda x: x["label"] != "Disagreement")
        self.dataset["train"] = self.dataset.pop("test")
