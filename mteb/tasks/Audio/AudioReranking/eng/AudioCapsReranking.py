from __future__ import annotations

import random

from datasets import Dataset

from mteb.abstasks.Audio.AbsTaskAudioReranking import AbsTaskAudioReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioCapsReranking(AbsTaskAudioReranking):
    metadata = TaskMetadata(
        name="AudioCapsReranking",
        description="Audio reranking using captions from the AudioCaps dataset",
        reference="https://audiocaps.github.io/",
        dataset={
            "path": "TwinkStart/AudioCaps",
            "revision": "8fc8b151149af779517aedfbf8c536160822bd70",
            "trust_remote_code": True,
        },
        type="AudioReranking",
        category="a2t",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2019-04-01", "2019-04-01"),
        domains=["General"],
        task_subtypes=["Audio Reranking"],
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

    # Column names in the dataset
    audio_column_name: str = "audio"
    caption_column_name: str = "caption"
    id_column_name: str = "audiocap_id"

    # For the reranking task
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"

    def load_data(self):
        """Load and prepare the AudioCaps dataset for reranking."""
        # Load the dataset
        dataset = self.metadata.load_dataset()

        # Keep only a subset of the test data if specified in eval_splits
        if any("[:" in split for split in self.metadata.eval_splits):
            for split in self.metadata.eval_splits:
                if "[:" in split:
                    base_split, limit = split.split("[:")
                    limit = int(limit.replace("%", "").replace("]", ""))
                    if limit < 100:
                        # Take only the specified percentage
                        dataset[base_split] = dataset[base_split].select(
                            range(len(dataset[base_split]) * limit // 100)
                        )

        # Transform the dataset for reranking
        reranking_dataset = {}

        for split in self.metadata.eval_splits:
            base_split = split.split("[")[0] if "[" in split else split

            reranking_dataset[split] = self._prepare_reranking_data(dataset[base_split])

        self.dataset = reranking_dataset

    def _prepare_reranking_data(self, dataset: Dataset) -> Dataset:
        """Transform the dataset to the reranking format.
        For each audio sample, we use it as the query, the same audio as the positive example,
        and random different audios as negative examples.
        """
        queries = []
        positives = []
        negatives = []

        # Group audio samples by their captions (to avoid pairing audios with similar captions)
        audio_by_caption = {}
        for i, item in enumerate(dataset):
            caption = item[self.caption_column_name]
            if caption not in audio_by_caption:
                audio_by_caption[caption] = []
            audio_by_caption[caption].append(i)

        all_indices = list(range(len(dataset)))

        for i, item in enumerate(dataset):
            # Use the current audio as query
            query = item[self.audio_column_name]

            # Use the same audio as the positive example
            positive = [item[self.audio_column_name]]

            # Sample 5-10 negative examples (audios with different captions)
            caption = item[self.caption_column_name]
            # Get indices of audios with different captions
            other_captions = [
                idx
                for c, indices in audio_by_caption.items()
                if c != caption
                for idx in indices
            ]

            # If not enough samples with different captions, use random samples
            if len(other_captions) < 5:
                other_indices = [idx for idx in all_indices if idx != i]
                other_captions = random.sample(
                    other_indices, min(10, len(other_indices))
                )
            else:
                other_captions = random.sample(
                    other_captions, min(10, len(other_captions))
                )

            # Get the negative examples
            negative = [dataset[idx][self.audio_column_name] for idx in other_captions]

            queries.append(query)
            positives.append(positive)
            negatives.append(negative)

        # Create the reranking dataset
        return Dataset.from_dict(
            {
                self.audio_query_column_name: queries,
                self.audio_positive_column_name: positives,
                self.audio_negative_column_name: negatives,
            }
        )
