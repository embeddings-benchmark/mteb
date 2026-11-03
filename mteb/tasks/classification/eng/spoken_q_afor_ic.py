from typing import Any

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
@inproceedings{shon2023sluephase2benchmarksuite,
  address = {Toronto, Canada},
  author = {Shon, Suwon and Arora, Siddhant and Lin, Chyi-Jiunn and Pasad, Ankita and Wu, Felix and Sharma, Roshan and Wu, Wei-Lun and Lee, Hung-yi and Livescu, Karen and Watanabe, Shinji},
  booktitle = {Proceedings of the 61st Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2023.acl-long.496},
  editor = {Rogers, Anna and Boyd-Graber, Jordan and Okazaki, Naoaki},
  month = {jul},
  pages = {8906--8937},
  publisher = {Association for Computational Linguistics},
  title = {{SLUE} {P}hase-2: A Benchmark Suite of Diverse Spoken Language Understanding Tasks},
  url = {https://aclanthology.org/2023.acl-long.496/},
  year = {2023},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True

    def dataset_transform(self, **kwargs: Any):
        self.dataset["train"] = self.dataset.pop("test")
