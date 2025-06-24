from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioReranking import AbsTaskAudioReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class VocalSoundAudioReranking(AbsTaskAudioReranking):
    """VocalSound dataset adapted for audio reranking task.

    The VocalSound dataset consists of over 21,000 crowdsourced recordings of human vocal sounds
    from 3,365 unique subjects across 6 categories: laughter, sighs, coughs, throat clearing,
    sneezes, and sniffs. The recordings include rich metadata such as speaker age, gender,
    native language, country, and health condition. This version is adapted for audio reranking
    where given a query vocal sound from one of the 6 categories, the task is to rank positive
    audio samples (same vocal sound type) higher than negative samples (different vocal sounds).

    Each query has 5 positive examples (same vocal sound category) and 20 negative examples
    (different vocal sound categories), creating a challenging vocal sound retrieval scenario
    with 25 total candidates per query. The dataset contains 360 queries providing robust
    evaluation across all 6 vocal sound categories with excellent speaker diversity.
    """

    metadata = TaskMetadata(
        name="VocalSoundAudioReranking",
        description="VocalSound dataset adapted for audio reranking. Given a query vocal sound from one of 6 categories, rank 5 relevant vocal samples higher than 20 irrelevant ones from different vocal sound types. Contains 360 queries across 6 vocal sound categories for robust evaluation.",
        reference="https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations",
        dataset={
            "path": "AdnanElAssadi/vocalsound-audio-reranking",
            "revision": "main",
        },
        type="AudioReranking",
        category="a2a", 
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2022-01-01", "2022-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotion Reranking"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{inproceedings,
  author = {Gong, Yuan and Yu, Jin and Glass, James},
  doi = {10.1109/ICASSP43922.2022.9746828},
  month = {05},
  pages = {151-155},
  title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
  year = {2022},
}
""",
        descriptive_stats={
            "n_samples": {"test": 300},
        },
    )

    # Column names from our preprocessed dataset
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"
