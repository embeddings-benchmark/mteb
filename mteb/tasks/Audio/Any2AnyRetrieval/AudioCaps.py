from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioCapsRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsRetrieval",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.",
        reference="https://allenai.org/data/arc",
        dataset={
            "path": "AudioLLMs/audiocaps_test",
            "revision": "fb42aac15212cbddd723fbbf04b6071b60a9f8fe",
        },
        type="Retrieval",
        category="s2s",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{clark2018think,
  author = {Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
  journal = {arXiv preprint arXiv:1803.05457},
  title = {Think you have solved question answering? try arc, the ai2 reasoning challenge},
  year = {2018},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        # prompt={"query": "Retrieve the answer to the question."},
    )

    audio_column_name: str = "context"
    text_column_name: str = "answer"
    id_column_name: str = "audiocap_id"

    def dataset_transform(self):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split]

            self.dataset[split] = self.dataset[split].add_column(
                "audiocap_id",
                [f"audiocap_{i}" for i in range(len(self.dataset[split]))],
            )
