from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioReranking import AbsTaskAudioReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class FSDnoisy18kAudioReranking(AbsTaskAudioReranking):
    """FSDnoisy18k dataset adapted for audio reranking task.

    FSDnoisy18k is an audio dataset collected with the aim of fostering the investigation of label noise in
    sound event classification. It contains audio clips unequally distributed among 20 sound classes.
    The audio clips are of variable length and can contain multiple sound events.
    This version is adapted for audio reranking where given a query audio from one of the 20 sound classes,
    the task is to rank positive audio samples (same class) higher than negative samples (different classes).

    Each query has 4 positive examples (same sound class) and 16 negative examples
    (different sound classes), creating a challenging noisy audio retrieval scenario
    with 20 total candidates per query. The dataset contains 200 queries providing robust
    evaluation across all 20 sound event categories, including handling of label noise.
    """

    metadata = TaskMetadata(
        name="FSDnoisy18kAudioReranking",
        description="FSDnoisy18k sound event dataset adapted for audio reranking. Given a query audio with potential label noise, rank 4 relevant audio samples higher than 16 irrelevant ones from different sound classes. Contains 200 queries across 20 sound event categories.",
        reference="https://zenodo.org/record/2529934",
        dataset={
            "path": "AdnanElAssadi/fsdnoisy18k-audio-reranking",
            "revision": "46ec066bfd2a78bae0b7ce71bbfbb479fba151c5",
        },
        type="AudioReranking",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2018-01-01", "2018-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Reranking"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{fonseca2019fsdnoisy18k,
  title     = {Learning Sound Event Classifiers from Web Audio with Noisy Labels},
  author    = {Fonseca, Eduardo and Plakal, Manoj and Ellis, Daniel P. W. and Font, Frederic and Favory, Xavier and Serra, Xavier},
  booktitle = {ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  year      = {2019},
  pages     = {21--25},
  organization = {IEEE}
}
}
""",
        descriptive_stats={
            "n_samples": {"test": 200},
        },
    )

    # Column names from our preprocessed dataset
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"
