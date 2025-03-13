from __future__ import annotations

import datasets

from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SugarCrepe(AbsTaskImageTextPairClassification):
    images_column_names = ["images"]
    texts_column_names = ["caption", "negative_caption"]

    metadata = TaskMetadata(
        name="SugarCrepe",
        description="Compositionality Evaluation of images to their captions.",
        reference="https://proceedings.neurips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html",
        dataset={
            "path": "yjkimstats/SUGARCREPE_fmt",
            "revision": "134abf9ade6a32f9fdae0e89022ff227a70b87e5",
        },
        type="Compositionality",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="text_acc",
        date=(
            "2022-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of data
        domains=["Encyclopaedic"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@article{hsieh2024sugarcrepe,
  title={Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ma, Zixian and Kembhavi, Aniruddha and Krishna, Ranjay},
  journal={Advances in neural information processing systems},
  volume={36},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 7511},
            "avg_character_length": {"test": 1},
        },
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])  # type: ignore
        self.dataset = datasets.DatasetDict({"test": self.dataset["train"]})
        self.dataset_transform()
        self.data_loaded = True
