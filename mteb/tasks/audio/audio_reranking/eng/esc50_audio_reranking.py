from mteb.abstasks.audio.abs_task_audio_reranking import AbsTaskAudioReranking
from mteb.abstasks.task_metadata import TaskMetadata


class ESC50AudioReranking(AbsTaskAudioReranking):
    """ESC-50 dataset adapted for audio reranking task.

    The Environmental Sound Classification 50 (ESC-50) dataset consists of 2000 environmental audio recordings
    suitable for benchmarking methods for environmental sound classification. This version is adapted for
    audio reranking where given a query audio from one of 50 environmental sound classes, the task is to
    rank positive audio samples (same class) higher than negative samples (different classes).

    Each query has 5 positive examples (same environmental sound class) and 16 negative examples
    (different environmental sound classes), creating a challenging audio-to-audio retrieval scenario
    with 21 total candidates per query. The dataset contains 200 queries providing robust evaluation
    across all 50 environmental sound categories.
    """

    metadata = TaskMetadata(
        name="ESC50AudioReranking",
        description="ESC-50 environmental sound dataset adapted for audio reranking. Given a query audio of environmental sounds, rank 5 relevant audio samples higher than 16 irrelevant ones from different sound classes. Contains 200 queries across 50 environmental sound categories for robust evaluation.",
        reference="https://github.com/karolpiczak/ESC-50",
        dataset={
            "path": "AdnanElAssadi/esc-50-audio-reranking",
            "revision": "5e0a9d84912522db2eb6b825cbfd09fb20df2a61",
        },
        type="AudioReranking",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2015-01-01", "2015-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Reranking"],
        license="cc-by-3.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015esc,
  author = {Piczak, Karol J},
  booktitle = {Proceedings of the 23rd ACM international conference on Multimedia},
  pages = {1015--1018},
  title = {ESC: Dataset for Environmental Sound Classification},
  year = {2015},
}
""",
    )

    # Column names from our preprocessed dataset
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"
