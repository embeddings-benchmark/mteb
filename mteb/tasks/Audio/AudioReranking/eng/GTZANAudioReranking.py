from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioReranking import AbsTaskAudioReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class GTZANAudioReranking(AbsTaskAudioReranking):
    """GTZAN music genre dataset adapted for audio reranking task.

    The GTZAN dataset consists of 1000 audio tracks each 30 seconds long, containing 10 music genres:
    blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Each genre is
    represented by 100 tracks. This version is adapted for audio reranking where given a query audio
    from one of the 10 music genres, the task is to rank positive audio samples (same genre) higher
    than negative samples (different genres).

    Each query has 3 positive examples (same music genre) and 10 negative examples
    (different music genres), creating a focused music genre retrieval scenario
    with 13 total candidates per query. The dataset contains 150 queries providing comprehensive
    evaluation across all 10 music genres.
    """

    metadata = TaskMetadata(
        name="GTZANAudioReranking",
        description="GTZAN music genre dataset adapted for audio reranking. Given a query audio from one of 10 music genres, rank 3 relevant audio samples higher than 10 irrelevant ones from different genres. Contains 150 queries across 10 music genres for comprehensive evaluation.",
        reference="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification",
        dataset={
            "path": "AdnanElAssadi/gtzan-audio-reranking",
            "revision": "65acd91773161ed7f28d788b058802174f3324d9",
        },
        type="AudioReranking",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2001-01-01", "2001-12-31"),
        domains=["Music"],
        task_subtypes=["Music Genre Reranking"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{1021072,
  author = {Tzanetakis, G. and Cook, P.},
  doi = {10.1109/TSA.2002.800560},
  journal = {IEEE Transactions on Speech and Audio Processing},
  keywords = {Humans;Music information retrieval;Instruments;Computer science;Multiple signal classification;Signal analysis;Pattern recognition;Feature extraction;Wavelet analysis;Cultural differences},
  number = {5},
  pages = {293-302},
  title = {Musical genre classification of audio signals},
  volume = {10},
  year = {2002},
}
""",
        descriptive_stats={
            "n_samples": {"test": 150},
        },
    )

    # Column names from our preprocessed dataset
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"
