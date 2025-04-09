from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata

# ASSUMED VOXCELEB IN CLASSIFICATION TASK WAS ACCURATE.


class VoxCelebClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="VoxCelebClustering",
        description="Clustering task based on the VoxCeleb dataset for sentiment analysis, clustering by positive/negative sentiment.",
        reference="https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
        dataset={
            "path": "DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
            "revision": "554ad4367e98b7c6f4d4d9756dc6bbdf345e042e",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="cluster_accuracy",
        date=("2024-06-27", "2024-06-28"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Sentiment Clustering"],
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
    )

    def dataset_transform(self):
        ## remove disagreement data
        self.dataset = self.dataset.filter(lambda x: x["label"] != "Disagreement")
        self.dataset["train"] = self.dataset.pop("test")
