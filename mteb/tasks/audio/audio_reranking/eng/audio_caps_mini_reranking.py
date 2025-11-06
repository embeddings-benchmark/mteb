from mteb.abstasks.audio.abs_task_adio_reranking import AbsTaskAudioReranking
from mteb.abstasks.task_metadata import TaskMetadata


class AudioCapsMiniReranking(AbsTaskAudioReranking):
    """A smaller version of AudioCapsReranking that uses a preprocessed dataset.
    This dataset is much smaller and already formatted for reranking,
    avoiding the need to download and process the full AudioCaps dataset.
    """

    metadata = TaskMetadata(
        name="AudioCapsMiniReranking",
        description="A smaller subset of AudioCaps dataset preprocessed for audio reranking",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "AdnanElAssadi/audiocaps-mini",
            "revision": "d4dc581109fd9ea6b8ed2c4a85075a5ec910cfda",
        },
        type="AudioReranking",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],  # The mini dataset has only one split
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2019-04-01", "2019-04-01"),
        domains=["Speech"],
        task_subtypes=["Environment Sound Reranking"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kim2019audiocaps,
  author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages = {119--132},
  title = {AudioCaps: Generating captions for audios in the wild},
  year = {2019},
}
""",
    )

    # The column names from our preprocessed dataset
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"
