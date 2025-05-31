from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioReranking import AbsTaskAudioReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


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
            # Replace YOUR_USERNAME with your actual Hugging Face username
            "path": "AdnanElAssadi/audiocaps-mini",
            "revision": "d4dc581109fd9ea6b8ed2c4a85075a5ec910cfda",  # Use the latest revision
        },
        type="AudioReranking",
        category="a2t",
        modalities=["audio"],
        eval_splits=["test"],  # The mini dataset has only one split
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2019-04-01", "2019-04-01"),
        domains=["General"],
        task_subtypes=["AudioReranking"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{kim2019audiocaps,
  title={AudioCaps: Generating captions for audios in the wild},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={119--132},
  year={2019}
}""",
    )

    # The column names from our preprocessed dataset
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"

    def load_data(self):
        """Load the preprocessed dataset that's already in the reranking format.
        No transformation needed as the dataset is already prepared.
        """
        # Load the dataset directly - it's already in the correct format
        self.dataset = self.metadata.load_dataset()
