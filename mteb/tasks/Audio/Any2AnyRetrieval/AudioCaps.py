from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioCapsA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsA2TRetrieval",
        description="Natural language description for any kind of audio in the wild.",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "AudioLLMs/audiocaps_test",
            "revision": "fb42aac15212cbddd723fbbf04b6071b60a9f8fe",
        },
        type="Retrieval",
        category="a2t",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kim2019audiocaps,
  author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages = {119--132},
  title = {Audiocaps: Generating captions for audios in the wild},
  year = {2019},
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


class AudioCapsT2ARetrieval(AbsTaskAny2AnyRetrieval):
    """This reverses the normal AudioCaps dataset. Instead of audio queries to text corpus, this tests text queries to audio corpus."""

    metadata = TaskMetadata(
        name="AudioCapsT2ARetrieval",
        description="Natural language description for any kind of audio in the wild.",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "AudioLLMs/audiocaps_test",
            "revision": "fb42aac15212cbddd723fbbf04b6071b60a9f8fe",
        },
        type="Retrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kim2019audiocaps,
  author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages = {119--132},
  title = {Audiocaps: Generating captions for audios in the wild},
  year = {2019},
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
