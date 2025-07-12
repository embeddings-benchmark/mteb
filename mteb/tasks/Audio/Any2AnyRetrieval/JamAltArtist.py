from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class JamAltArtist(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="JamAltArtistA2ARetrieval",
        description="Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "jamendolyrics/jam-alt-lines",
            "revision": "main",
        },
        type="Retrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "fra-Latn", "deu-Latn", "spa-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-12-31"),
        domains=["Music"],
        task_subtypes=["Music Genre Classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{cifka-2024-jam-alt,
  author = {Ond{\v{r}}ej C{\'{\i}}fka and
Hendrik Schreiber and
Luke Miner and
Fabian{-}Robert St{\"{o}}ter},
  booktitle = {Proceedings of the 25th International Society for
Music Information Retrieval Conference},
  doi = {10.5281/ZENODO.14877443},
  pages = {737--744},
  publisher = {ISMIR},
  title = {Lyrics Transcription for Humans: {A} Readability-Aware Benchmark},
  url = {https://doi.org/10.5281/zenodo.14877443},
  year = {2024},
}

@inproceedings{syed-2025-mss-alt,
  author = {Jaza Syed and
Ivan Meresman-Higgs and
Ond{\v{r}}ej C{\'{\i}}fka and
Mark Sandler},
  booktitle = {2025 {IEEE} International Conference on Multimedia and Expo Workshops (ICMEW)},
  publisher = {IEEE},
  title = {Exploiting Music Source Separation for Automatic Lyrics Transcription with {Whisper}},
  year = {2025},
}
""",
        prompt={
            "query": "Retrieve all songs created by the same artist as the following audio clip: {query}"
        },
    )

    audio_column_name: str = "audio"  # audio clips for both corpus & query
    text_column_name: str = "artist"  # 74 unique artist names total
    id_column_name: str = "segment_id"

    default_query_modality: str = "audio"
    default_corpus_modality: str = "audio"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # load dataset with streaming
        self.dataset = datasets.load_dataset(
            **self.metadata_dict["dataset"], streaming=True
        )
        self.dataset_transform()

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        eval_splits = kwargs.get("eval_splits", self.metadata_dict["eval_splits"])

        for split_config in eval_splits:
            split = split_config.split("[")[0]

            if split not in self.dataset:
                continue

            dataset_split = self.dataset[split]

            # store the dataset split reference with slice config
            self.corpus[split] = {
                "dataset_split": dataset_split,
                "split_config": split_config,
            }
            self.queries[split] = {
                "dataset_split": dataset_split,
                "split_config": split_config,
            }
            self.relevant_docs[split] = {
                "dataset_split": dataset_split,
                "split_config": split_config,
            }

        self.data_loaded = True

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ):
        # if corpus/queries are still dataset references, process them now
        if isinstance(corpus, dict) and "dataset_split" in corpus:
            corpus = self._process_dataset_split(corpus)

        if isinstance(queries, dict) and "dataset_split" in queries:
            queries = self._process_dataset_split(queries)

        if isinstance(relevant_docs, dict) and "dataset_split" in relevant_docs:
            relevant_docs = self._process_dataset_split_for_qrels(relevant_docs)

        # call parent method with processed data
        return super()._evaluate_subset(
            retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
        )

    def _process_dataset_split(self, dataset_ref):
        # process a dataset split to extract actual audio/text
        # also implements balanced artist sampling (covers all unique artists but not every clip becomes a query)
        dataset_split = dataset_ref["dataset_split"]
        split_config = dataset_ref["split_config"]

        if "[" in split_config and "]" in split_config:
            slice_part = split_config.split("[")[1].split("]")[0]
            if ":" in slice_part:
                slice_parts = slice_part.split(":")
                start = int(slice_parts[0]) if slice_parts[0] else 0
                end = int(slice_parts[1]) if slice_parts[1] else None
                if start > 0:
                    dataset_split = dataset_split.skip(start)
                if end is not None:
                    take_count = end - start
                    dataset_split = dataset_split.take(take_count)
            else:
                index = int(slice_part)
                dataset_split = dataset_split.skip(index).take(1)

        # collect items by artist

        artist_items = defaultdict(list)

        # first pass: group by artist
        max_total_items = 500
        items_processed = 0

        for item in dataset_split:
            if max_total_items is not None and items_processed >= max_total_items:
                break
            artist = item["artist"]
            artist_items[artist].append(item)
            items_processed += 1

        # second pass: sample balanced items per artist
        result = {}
        items_per_artist = 4  # take 4 samples per artist for good coverage

        for artist, items in artist_items.items():
            selected_items = items[:items_per_artist]
            for idx, item in enumerate(selected_items):
                segment_id = f"{item['song_name']}_{artist}_{idx}"
                result[segment_id] = item["audio"]

        print(
            f"Balanced sampling: {len(artist_items)} artists, {len(result)} total samples"
        )
        return result

    def _process_dataset_split_for_qrels(self, dataset_ref):
        """Process a dataset split reference to extract qrels with balanced artist sampling."""
        dataset_split = dataset_ref["dataset_split"]
        split_config = dataset_ref["split_config"]

        if "[" in split_config and "]" in split_config:
            slice_part = split_config.split("[")[1].split("]")[0]
            if ":" in slice_part:
                slice_parts = slice_part.split(":")
                start = int(slice_parts[0]) if slice_parts[0] else 0
                end = int(slice_parts[1]) if slice_parts[1] else None
                if start > 0:
                    dataset_split = dataset_split.skip(start)
                if end is not None:
                    take_count = end - start
                    dataset_split = dataset_split.take(take_count)
            else:
                index = int(slice_part)
                dataset_split = dataset_split.skip(index).take(1)

        artist_items = defaultdict(list)

        max_total_items = 500
        items_processed = 0

        for item in dataset_split:
            if items_processed >= max_total_items:
                break
            artist = item["artist"]
            artist_items[artist].append(item)
            items_processed += 1

        # create segment mappings with balanced sampling
        artist_segments = defaultdict(list)
        segment_to_artist = {}
        items_per_artist = 4

        for artist, items in artist_items.items():
            selected_items = items[:items_per_artist]

            for idx, item in enumerate(selected_items):
                segment_id = f"{item['song_name']}_{artist}_{idx}"
                artist_segments[artist].append(segment_id)
                segment_to_artist[segment_id] = artist

        # build qrels -- each audio clip query maps to all clips from same artist
        qrels = {}
        for segment_id in segment_to_artist.keys():
            artist = segment_to_artist[segment_id]
            qrels[segment_id] = {}
            for related_segment_id in artist_segments[artist]:
                qrels[segment_id][related_segment_id] = 1

        return qrels
