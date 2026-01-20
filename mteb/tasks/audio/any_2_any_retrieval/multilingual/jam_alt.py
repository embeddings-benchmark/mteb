from datasets import Audio, DatasetDict, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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

    # Cast audio column once if needed
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=True))

    # Group indices by language
    lang_indices = {lang: [] for lang in langs}
    for idx, row in enumerate(ds):
        lang = row["song_language"]
        if lang in lang_indices:
            lang_indices[lang].append(idx)

    for lang, indices in lang_indices.items():
        if not indices:
            continue

        # Use select to avoid copying data
        lang_ds = ds.select(indices)

        # Generate IDs using map (avoids creating new dataset)
        def add_corpus_id(example, idx):
            example["id"] = f"corpus-{example['song_name']}{example['line_indices'][0]}"
            return example

        def add_query_id(example, idx):
            example["id"] = f"query-{example['song_name']}{example['line_indices'][0]}"
            return example

        # Create corpus
        corpus_ds = lang_ds.map(add_corpus_id, with_indices=True)
        corpus_ds = corpus_ds.select_columns(["id", corpus_column])
        corpus[lang][split] = corpus_ds

        # Create queries
        query_ds = lang_ds.map(add_query_id, with_indices=True)
        query_ds = query_ds.select_columns(["id", query_column])
        queries[lang][split] = query_ds

        # Build qrels efficiently
        qrels = {}
        qrel_groups = {}

        # Group by qrels_column
        for row in lang_ds:
            qrel_key = row[qrels_column]
            query_id = f"query-{row['song_name']}{row['line_indices'][0]}"
            corpus_id = f"corpus-{row['song_name']}{row['line_indices'][0]}"

            if qrel_key not in qrel_groups:
                qrel_groups[qrel_key] = {"queries": set(), "corpus": set()}

            qrel_groups[qrel_key]["queries"].add(query_id)
            qrel_groups[qrel_key]["corpus"].add(corpus_id)

        # Create cross-product within each group
        for group in qrel_groups.values():
            for query_id in group["queries"]:
                if query_id not in qrels:
                    qrels[query_id] = {}
                for corpus_id in group["corpus"]:
                    qrels[query_id][corpus_id] = 1

        relevant_docs[lang][split] = qrels

    corpus = DatasetDict({lang: DatasetDict(splits) for lang, splits in corpus.items()})
    queries = DatasetDict(
        {lang: DatasetDict(splits) for lang, splits in queries.items()}
    )
    relevant_docs = DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class JamAltArtist(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JamAltArtistA2ARetrieval",
        description="Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "mteb/jam-alt-lines",
            "revision": "e2d97afa7333eb489f1d451f76079d921be7a68f",
            "name": "pure",
        },
        type="Any2AnyRetrieval",
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
            splits=self.metadata.eval_splits,
            langs=self.hf_subsets,
            dataset_args=self.metadata.dataset,
            query_column="audio",
            corpus_column="audio",
            qrels_column="artist",
        )

        self.data_loaded = True


class JamAltLyricsT2A(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JamAltLyricT2ARetrieval",
        description="From textual lyrics (query), retrieve corresponding audio clips of songs from the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "mteb/jam-alt-lines",
            "revision": "e2d97afa7333eb489f1d451f76079d921be7a68f",
            "name": "pure",
        },
        type="Any2AnyRetrieval",
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
            splits=self.metadata.eval_splits,
            langs=self.hf_subsets,
            dataset_args=self.metadata.dataset,
            query_column="text",
            corpus_column="audio",
            qrels_column="text",
        )

        self.data_loaded = True


class JamAltLyricsA2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JamAltLyricA2TRetrieval",
        description="From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/jamendolyrics/jam-alt-lines",
        dataset={
            "path": "mteb/jam-alt-lines",
            "revision": "e2d97afa7333eb489f1d451f76079d921be7a68f",
            "name": "pure",
        },
        type="Any2AnyRetrieval",
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
            splits=self.metadata.eval_splits,
            langs=self.hf_subsets,
            dataset_args=self.metadata.dataset,
            query_column="audio",
            corpus_column="text",
            qrels_column="text",
        )

        self.data_loaded = True
