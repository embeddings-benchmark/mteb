from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class SciMMIR(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="SciMMIR",
        description="SciMMIR.",
        reference="https://huggingface.co/datasets/m-a-p/SciMMIR",
        dataset={
            "path": "mteb/SciMMIR",
            "revision": "2a10b6b1f93cac7084aa809512749272cf0a893a",
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
        bibtex_citation=r"""
@inproceedings{wu2024scimmirbenchmarkingscientificmultimodal,
  address = {Bangkok, Thailand},
  author = {Wu, Siwei and Li, Yizhi and Zhu, Kang and Zhang, Ge and Liang, Yiming and Ma, Kaijing and Xiao, Chenghao and Zhang, Haoran and Yang, Bohao and Chen, Wenhu and Huang, Wenhao and Al Moubayed, Noura and Fu, Jie and Lin, Chenghua},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2024},
  doi = {10.18653/v1/2024.findings-acl.746},
  editor = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  month = {aug},
  pages = {12560--12574},
  publisher = {Association for Computational Linguistics},
  title = {{S}ci{MMIR}: Benchmarking Scientific Multi-modal Information Retrieval},
  url = {https://aclanthology.org/2024.findings-acl.746/},
  year = {2024},
}
""",
    )

    label_column_name: str = "class"

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
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

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return [
            "a figure of results",
            "a figure of an illustration",
            "a figure of an architecture",
            "a table of parameters",
            "a table of results",
        ]
