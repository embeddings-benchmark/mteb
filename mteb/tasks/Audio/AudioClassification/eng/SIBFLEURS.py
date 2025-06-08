from __future__ import annotations

from datasets import Dataset, DatasetDict

from mteb.abstasks.Audio.AbsTaskAudioClassification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class SIBFLEURSMultilingualClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SIBFLEURS",
        description="Topic Classification for multilingual audio dataset",
        reference="https://huggingface.co/datasets/WueNLP/sib-fleurs",
        dataset={
            "path": "WueNLP/sib-fleurs",
            "name": "eng_Latn",
            "revision": "f00a6bbc6b3e3866600f838736295dd09b393902",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2024-12-09",
            "2024-12-13",
        ),
        domains=[
            "Encyclopaedic"
        ],  # original FLORES-101 dataset is read-out wikipedia corpus
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{schmidt2025fleursslumassivelymultilingualbenchmark,
  archiveprefix = {arXiv},
  author = {Fabian David Schmidt and Ivan Vulić and Goran Glavaš and David Ifeoluwa Adelani},
  eprint = {2501.06117},
  primaryclass = {cs.CL},
  title = {Fleurs-SLU: A Massively Multilingual Benchmark for Spoken Language Understanding},
  url = {https://arxiv.org/abs/2501.06117},
  year = {2025},
}
""",
        descriptive_stats={
            "n_samples": {"test": 177},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "category"
    samples_per_label: int = 10

    def dataset_transform(self):
        ## flatten multi element entries in the dataset for samples with multiple speakers.
        dataset = {}
        for split in self.dataset.keys():
            data = self.dataset[split]
            transformed_data = {
                "audio": [],
                "category": [],
                "gender": [],
                "speaker_id": [],
                "fleurs_id": [],
            }
            for d in data:
                for i in range(len(d["audio"])):
                    # a number of features are kept for later usage in other tasks
                    transformed_data["audio"].append(d["audio"][i])
                    transformed_data["category"].append(d["category"])
                    transformed_data["gender"].append(d["gender"][i])
                    transformed_data["speaker_id"].append(d["speaker_id"][i])
                    transformed_data["fleurs_id"].append(d["fleurs_id"])
            dataset[split] = Dataset.from_dict(transformed_data)
        self.dataset = DatasetDict(dataset)
