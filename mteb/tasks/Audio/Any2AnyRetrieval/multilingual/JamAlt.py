from __future__ import annotations

import polars as pl
from datasets import Audio, DatasetDict, load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_LANGS = {
    "en": ["eng-Latn"],
    "fr": ["fra-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
}


def _load_jam_alt_data(
    splits,
    langs,
    dataset_args,
    query_column: str,
    corpus_column: str,
    qrels_column: str,
    **kwargs,
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}

    split = "test"
    ds = load_dataset(**dataset_args, split=split)

    # Convert to Polars DataFrame for fast filtering
    df = pl.DataFrame(ds.to_pandas())

    for lang in langs:
        # Fast filtering with Polars
        lang_df = df.filter(pl.col("song_language") == lang)

        # Create corpus data
        corpus_data = lang_df.with_columns(
            [
                (
                    pl.lit("corpus-")
                    + pl.col("song_name")
                    + pl.col("line_indices").list.get(0).cast(pl.Utf8)
                ).alias("id"),
                pl.lit(corpus_column).alias("modality"),
            ]
        ).select(["id", "modality", corpus_column])

        # Convert back to HF dataset and recast audio column if needed
        lang_corpus = ds.from_pandas(corpus_data.to_pandas())
        if corpus_column == "audio":
            lang_corpus = lang_corpus.cast_column("audio", Audio(decode=True))
        corpus[lang][split] = lang_corpus

        # Create queries data
        query_data = lang_df.with_columns(
            [
                (
                    pl.lit("query-")
                    + pl.col("song_name")
                    + pl.col("line_indices").list.get(0).cast(pl.Utf8)
                ).alias("id"),
                pl.lit(query_column).alias("modality"),
            ]
        ).select(["id", "modality", query_column])

        # Convert back to HF dataset and recast audio column if needed
        lang_query = ds.from_pandas(query_data.to_pandas())
        if query_column == "audio":
            lang_query = lang_query.cast_column("audio", Audio(decode=True))
        queries[lang][split] = lang_query

        # Build qrels efficiently with Polars
        relevant_docs[lang][split] = {}

        # Create query and corpus IDs
        lang_df_with_ids = lang_df.with_columns(
            [
                (
                    pl.lit("query-")
                    + pl.col("song_name")
                    + pl.col("line_indices").list.get(0).cast(pl.Utf8)
                ).alias("query_id"),
                (
                    pl.lit("corpus-")
                    + pl.col("song_name")
                    + pl.col("line_indices").list.get(0).cast(pl.Utf8)
                ).alias("corpus_id"),
            ]
        )

        # Group by qrels_column for efficient matching
        qrel_groups = lang_df_with_ids.group_by(qrels_column).agg(
            [pl.col("query_id").unique(), pl.col("corpus_id").unique()]
        )

        # Build qrels from grouped data
        for row in qrel_groups.iter_rows(named=True):
            query_ids = row["query_id"]
            corpus_ids = row["corpus_id"]

            # Create all combinations within this group
            for query_id in query_ids:
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}

                for corpus_id in corpus_ids:
                    relevant_docs[lang][split][query_id][corpus_id] = 1

    corpus = DatasetDict({lang: DatasetDict(splits) for lang, splits in corpus.items()})
    queries = DatasetDict(
        {lang: DatasetDict(splits) for lang, splits in queries.items()}
    )
    relevant_docs = DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class JamAltArtist(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="JamAltArtistA2ARetrieval",
        description="Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "jamendolyrics/jam-alt-lines",
            "revision": "11dc96be3bbefd4eb49a467825d7d3d3808105d7",
            "name": "pure",
        },
        type="Any2AnyMultilingualRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
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
""",
        prompt={
            "query": "Retrieve all songs created by the same artist as the following audio clip: {query}"
        },
    )
    skip_first_result = True

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_jam_alt_data(
            splits=self.metadata_dict["eval_splits"],
            langs=self.hf_subsets,
            dataset_args=self.metadata_dict["dataset"],
            query_column="audio",
            corpus_column="audio",
            qrels_column="artist",
        )

        self.data_loaded = True


class JamAltLyricsT2A(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="JamAltLyricT2ARetrieval",
        description="From textual lyrics (query), retrieve corresponding audio clips of songs from the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "jamendolyrics/jam-alt-lines",
            "revision": "main",
        },
        type="Any2AnyMultilingualRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-12-31"),
        domains=["Music"],
        task_subtypes=["Song Lyrics Retrieval"],
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
""",
        prompt={"query": "Retrieve audio clip for the lyrics: {query}"},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_jam_alt_data(
            splits=self.metadata_dict["eval_splits"],
            langs=self.hf_subsets,
            dataset_args=self.metadata_dict["dataset"],
            query_column="text",
            corpus_column="audio",
            qrels_column="text",
        )

        self.data_loaded = True


class JamAltLyricsA2T(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="JamAltLyricA2TRetrieval",
        description="From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "jamendolyrics/jam-alt-lines",
            "revision": "main",
        },
        type="Any2AnyMultilingualRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-12-31"),
        domains=["Music"],
        task_subtypes=["Song Lyrics Retrieval"],
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
""",
        prompt={
            "query": "Retrieve textual lyric for the audio clips of songs: {query}"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_jam_alt_data(
            splits=self.metadata_dict["eval_splits"],
            langs=self.hf_subsets,
            dataset_args=self.metadata_dict["dataset"],
            query_column="audio",
            corpus_column="text",
            qrels_column="text",
        )

        self.data_loaded = True
