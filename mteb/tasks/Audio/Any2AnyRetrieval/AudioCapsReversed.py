from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioCapsReversed(AbsTaskAny2AnyRetrieval):
    """This reverses the normal AudioCaps dataset. Instead of audio queries to text corpus, this tests text queries to audio corpus."""

    metadata = TaskMetadata(
        name="AudioCapsReversed",
        description="Test task for any2any retrieval: text queries to find audio corpus",
        reference="https://allenai.org/data/arc",
        dataset={
            "path": "AudioLLMs/audiocaps_test",
            "revision": "fb42aac15212cbddd723fbbf04b6071b60a9f8fe",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text", "audio"],  # both modalities used
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
@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
    )

    audio_column_name: str = "context"
    text_column_name: str = "answer"
    id_column_name: str = "audiocap_id"

    # reverse the default modalities
    default_query_modality: str = "text"  # queries will be text
    default_corpus_modality: str = "audio"  # corpus will be audio

    def dataset_transform(self):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split]

            self.dataset[split] = self.dataset[split].add_column(
                "audiocap_id",
                [f"audiocap_{i}" for i in range(len(self.dataset[split]))],
            )
