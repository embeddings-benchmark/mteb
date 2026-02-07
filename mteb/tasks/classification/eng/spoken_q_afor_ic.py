from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SpokenQAForIC(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpokenQAForIC",
        description="SpokenQA dataset reformulated as Intent Classification (IC) task",
        reference="https://huggingface.co/datasets/DynamicSuperb/SpokenQA_SLUE",
        dataset={
            "path": "mteb/SpokenQA_SLUE",
            "revision": "97eb2287a0c881538cee9f5db415e80111d96a31",
        },
        type="AudioClassification",
        category="a2c",
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
        bibtex_citation=r"""
@misc{shon2023sluephase2benchmarksuite,
  archiveprefix = {arXiv},
  author = {Suwon Shon and Siddhant Arora and Chyi-Jiunn Lin and Ankita Pasad and Felix Wu and Roshan Sharma and Wei-Lun Wu and Hung-Yi Lee and Karen Livescu and Shinji Watanabe},
  eprint = {2212.10525},
  primaryclass = {cs.CL},
  title = {SLUE Phase-2: A Benchmark Suite of Diverse Spoken Language Understanding Tasks},
  url = {https://arxiv.org/abs/2212.10525},
  year = {2023},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True

    def dataset_transform(self):
        self.dataset["train"] = self.dataset.pop("test")
