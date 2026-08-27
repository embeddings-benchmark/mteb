from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LibriSpeechSpeakerA2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LibriSpeechSpeakerA2ARetrieval",
        description=(
            "Audio-to-audio speaker retrieval on LibriSpeech test-clean. "
            "Queries and corpus are disjoint utterances drawn from all 40 "
            "speakers in the test-clean split (200 queries / 400 corpus "
            "docs, 5 queries and 10 corpus docs per speaker); relevance is "
            "same-speaker membership, so a model must retrieve other "
            "recordings of the same person's voice rather than match on "
            "textual content."
        ),
        reference="https://www.openslr.org/12",
        dataset={
            "path": "yaswanth169/LibriSpeech-Speaker-A2ARetrieval",
            "revision": "ee15cd3b2769c34b8f4ddfe9022093c64cf82413",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2015-01-01", "2015-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{panayotov2015librispeech,
  author = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  organization = {IEEE},
  pages = {5206--5210},
  title = {Librispeech: An {ASR} corpus based on public domain audio books},
  year = {2015},
}
""",
        prompt={"query": "Retrieve other recordings of the same speaker."},
        is_beta=True,
    )
