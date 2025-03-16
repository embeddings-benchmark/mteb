from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpokenQAforIC(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SpokenQAforIC",
        description="SpokenQA dataset reformulated as Intent Classification (IC) task",
        reference="https://huggingface.co/datasets/DynamicSuperb/SpokenQA_SLUE",
        dataset={
            "path": "DynamicSuperb/SpokenQA_SLUE",
            "revision": "191367d68255d7bd50928c869690da15961666fd",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-11-18", "2023-11-18"),
        domains=["Spoken"],
        task_subtypes=["Intent Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="multiple",
        bibtex_citation="""@misc{shon2023sluephase2benchmarksuite,
            title={SLUE Phase-2: A Benchmark Suite of Diverse Spoken Language Understanding Tasks}, 
            author={Suwon Shon and Siddhant Arora and Chyi-Jiunn Lin and Ankita Pasad and Felix Wu and Roshan Sharma and Wei-Lun Wu and Hung-Yi Lee and Karen Livescu and Shinji Watanabe},
            year={2023},
            eprint={2212.10525},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2212.10525}, 
        }""",
        descriptive_stats={
            "n_samples": {"train": 6121},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 10
    is_cross_validation: bool = True

    def dataset_transform(self):
        ## required to run the dataloader for cross-validation
        import torch

        torch.multiprocessing.set_sharing_strategy("file_system")
        #########################################################

        self.dataset["train"] = self.dataset.pop("test")
