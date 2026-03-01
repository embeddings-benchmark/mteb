from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VocalSoundAudioReranking(AbsTaskRetrieval):
    """VocalSound dataset adapted for audio reranking task.

    The VocalSound dataset consists of recordings of human vocal sounds across 6 categories:
    laughter, sighs, coughs, throat clearing, sneezes, and sniffs. The recordings include rich
    metadata such as speaker age, gender, native language, country, and health condition. This
    version is adapted for audio reranking where given a query vocal sound from one of the 6 categories,
    the task is to rank positive audio samples (same vocal sound type) higher than negative
    samples (different vocal sounds).

    Each query has 4 positive examples (same vocal sound category) and 16 negative examples
    (different vocal sound categories), creating a challenging vocal sound retrieval scenario
    with 20 total candidates per query. The dataset contains 198 queries providing robust
    evaluation across all 6 vocal sound categories with excellent speaker diversity.
    """

    metadata = TaskMetadata(
        name="VocalSoundAudioReranking",
        description="VocalSound dataset adapted for audio reranking. Given a query vocal sound from one of 6 categories, rank 4 relevant vocal samples higher than 16 irrelevant ones from different vocal sound types. Contains 198 queries across 6 vocal sound categories for robust evaluation.",
        reference="https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations",
        dataset={
            "path": "mteb/VocalSoundAudioReranking",
            "revision": "d44884963d55578a13d6da80dfaca0f7f0971fe7",
        },
        type="AudioReranking",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2022-01-01", "2022-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotion Reranking"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Gong_2022,
  author = {Gong, Yuan and Yu, Jin and Glass, James},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/icassp43922.2022.9746828},
  month = may,
  publisher = {IEEE},
  title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
  url = {http://dx.doi.org/10.1109/ICASSP43922.2022.9746828},
  year = {2022},
}
""",
    )
