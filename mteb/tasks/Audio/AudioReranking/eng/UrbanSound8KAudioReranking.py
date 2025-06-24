from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioReranking import AbsTaskAudioReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class UrbanSound8KAudioReranking(AbsTaskAudioReranking):
    """UrbanSound8K dataset adapted for audio reranking task.

    The UrbanSound8K dataset contains 8732 labeled sound excerpts (≤4s) of urban sounds from 10 classes:
    air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer,
    siren, and street_music. This version is adapted for audio reranking where given a query audio from
    one of the 10 urban sound classes, the task is to rank positive audio samples (same class) higher
    than negative samples (different classes).

    Each query has 4 positive examples (same urban sound class) and 16 negative examples
    (different urban sound classes), creating a focused urban audio retrieval scenario
    with 20 total candidates per query. The dataset contains 200 queries providing comprehensive
    evaluation across all 10 urban sound categories.
    """

    metadata = TaskMetadata(
        name="UrbanSound8KAudioReranking",
        description="UrbanSound8K urban sound dataset adapted for audio reranking. Given a query audio of urban sounds, rank 4 relevant audio samples higher than 16 irrelevant ones from different urban sound classes. Contains 200 queries across 10 urban sound categories for comprehensive evaluation.",
        reference="https://urbansounddataset.weebly.com/urbansound8k.html",
        dataset={
            "path": "AdnanElAssadi/urbansound8k-audio-reranking",
            "revision": "71a0190434dcd2ed2bf9bfc5d850a83b94759255",
        },
        type="AudioReranking",
        category="a2a", 
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2014-11-01", "2014-11-03"),
        domains=["Spoken"],
        task_subtypes=["Environment Sound Reranking"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{Salamon2014ADA,Add commentMore actions
  author = {Justin Salamon and Christopher Jacoby and Juan Pablo Bello},
  journal = {Proceedings of the 22nd ACM international conference on Multimedia},
  title = {A Dataset and Taxonomy for Urban Sound Research},
  url = {https://api.semanticscholar.org/CorpusID:207217115},
  year = {2014},
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
