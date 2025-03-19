from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SciMMIR(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="SciMMIR",
        description="SciMMIR.",
        reference="https://huggingface.co/datasets/m-a-p/SciMMIR",
        dataset={
            "path": "m-a-p/SciMMIR",
            "revision": "eea276dc58c52eab33e9476acb137ff5530b78e9",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-05-01", "2023-10-30"),
        domains=["Academic"],
        task_subtypes=["Caption Pairing", "Rendered Texts Understanding"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""\
@misc{wu2024scimmirbenchmarkingscientificmultimodal,
      title={SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval},
      author={Siwei Wu and Yizhi Li and Kang Zhu and Ge Zhang and Yiming Liang and Kaijing Ma and Chenghao Xiao and Haoran Zhang and Bohao Yang and Wenhu Chen and Wenhao Huang and Noura Al Moubayed and Jie Fu and Chenghua Lin},
      year={2024},
      eprint={2401.13478},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2401.13478},
}""",
        descriptive_stats={
            "n_samples": {"test": 16263},
            "avg_character_length": {"test": 0},
        },
    )

    label_column_name: str = "class"

    def dataset_transform(self):
        class_code = {
            "fig_result": 0,
            "fig_illustration": 1,
            "fig_architecture": 2,
            "table_parameter": 3,
            "table_result": 4,
        }
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].map(
                lambda example: {
                    "image": example["image"],
                    "class": class_code[example[self.label_column_name]],
                }
            )

    def get_candidate_labels(self) -> list[str]:
        return [
            "a figure of results",
            "a figure of an illustration",
            "a figure of an architecture",
            "a table of parameters",
            "a table of results",
        ]
