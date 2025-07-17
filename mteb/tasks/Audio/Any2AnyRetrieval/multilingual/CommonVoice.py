from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_LANGS = {
    "en": ["eng-Latn"],
    "fr": ["fra-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
}


def _load_common_voice_data(
    splits,
    langs,
    dataset_args,
    query_column: str,
    corpus_column: str,
    **kwargs,
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}

    split = "test"

    for lang in langs:
        lang_data = load_dataset(**dataset_args, split=split, name=lang)

        # Create corpus data
        corpus_modality = "audio"
        if corpus_column == "sentence":
            corpus_modality = "text"

        lang_corpus = lang_data.map(
            lambda x: {
                "id": "c-" + x["path"],
                "text": None,
                "modality": corpus_modality,
                corpus_modality: x[corpus_column],
            },
            remove_columns=[
                "client_id",
                "path",
            ],
        )

        corpus[lang][split] = lang_corpus

        # Create queries data
        query_modality = "audio"
        if query_column == "sentence":
            query_modality = "text"

        lang_query = lang_data.map(
            lambda x: {
                "id": "q-" + x["path"],
                "text": None,
                "modality": query_modality,
                query_modality: x[query_column],
            },
            remove_columns=[
                "client_id",
                "path",
            ],
        )

        queries[lang][split] = lang_query

        # Build qrels efficiently with Polars
        relevant_docs[lang][split] = {}

        # Create query and corpus IDs
        lang_query = lang_data.map(
            lambda x: {
                "id": "q-" + x["path"],
                "text": None,
                "modality": query_modality,
                query_modality: x[query_column],
            },
            remove_columns=[
                "client_id",
                "path",
            ],
        )

        # Build qrels from grouped data
        for row in lang_data:
            query_id = "q-" + row["path"]
            corpus_id = "c-" + row["path"]

            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}

            relevant_docs[lang][split][query_id][corpus_id] = 1

    corpus = DatasetDict({lang: DatasetDict(splits) for lang, splits in corpus.items()})
    queries = DatasetDict(
        {lang: DatasetDict(splits) for lang, splits in queries.items()}
    )
    relevant_docs = DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class CommonVoice17T2A(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17T2ARetrieval",
        description="From textual lyrics (query), retrieve corresponding audio clips of songs from the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
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

        self.corpus, self.queries, self.relevant_docs = _load_common_voice_data(
            splits=self.metadata_dict["eval_splits"],
            langs=self.hf_subsets,
            dataset_args=self.metadata_dict["dataset"],
            query_column="text",
            corpus_column="audio",
            qrels_column="text",
        )

        self.data_loaded = True


class CommonVoice17A2T(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17A2TRetrieval",
        description="From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset",
        reference="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0",
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

        self.corpus, self.queries, self.relevant_docs = _load_common_voice_data(
            splits=self.metadata_dict["eval_splits"],
            langs=self.hf_subsets,
            dataset_args=self.metadata_dict["dataset"],
            query_column="audio",
            corpus_column="text",
        )

        self.data_loaded = True
