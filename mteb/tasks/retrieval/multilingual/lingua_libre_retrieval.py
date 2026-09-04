from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/LinguaLibre-word-retrieval"
_DATASET_REVISION = "92576496cdf11e39315ec3c79e69da634177e13a"

# Scripts are taken from the recordings themselves rather than assumed from the language.
# South Levantine Arabic is romanised in this collection, while Moroccan Arabic is not.
_LANGUAGES = {
    "abl": ["abl-Latn"],
    "ace": ["ace-Latn"],
    "ajp": ["ajp-Latn"],
    "ary": ["ary-Arab"],
    "atj": ["atj-Latn"],
    "ban": ["ban-Latn"],
    "bar": ["bar-Latn"],
    "bci": ["bci-Latn"],
    "bcl": ["bcl-Latn"],
    "bew": ["bew-Latn"],
    "bik": ["bik-Latn"],
    "bjn": ["bjn-Latn"],
    "bkr": ["bkr-Latn"],
    "blk": ["blk-Mymr"],
    "bqr": ["bqr-Latn"],
    "btm": ["btm-Latn"],
    "bug": ["bug-Latn"],
    "cor": ["cor-Latn"],
    "dtp": ["dtp-Latn"],
    "ext": ["ext-Latn"],
    "fon": ["fon-Latn"],
    "gcf": ["gcf-Latn"],
    "gcr": ["gcr-Latn"],
    "gsw": ["gsw-Latn"],
    "hat": ["hat-Latn"],
    "jax": ["jax-Latn"],
    "kaa": ["kaa-Latn"],
    "ken": ["ken-Latn"],
    "kok": ["kok-Deva"],
    "kur": ["kur-Latn"],
    "kvb": ["kvb-Latn"],
    "lbx": ["lbx-Latn"],
    "ljp": ["ljp-Latn"],
    "mad": ["mad-Latn"],
    "mak": ["mak-Latn"],
    "min": ["min-Latn"],
}

_BIBTEX = r"""
@misc{lingualibre,
  author = {{Lingua Libre contributors}},
  howpublished = {\url{https://lingualibre.org}},
  title = {Lingua Libre},
  year = {2026},
}
"""

_DESCRIPTION = (
    "Single words read aloud by volunteers, paired with the written word, in 36 languages "
    "that no other audio task in mteb reaches. Lingua Libre is a Wikimedia project and the "
    "recordings are hosted on Wikimedia Commons."
)

_CONSTRUCTION = (
    "Commons files these under one category per language keyed by ISO 639-3, and the word "
    "and speaker both sit in the filename, so the label needs no annotation. Entries "
    "beginning with a non-letter are dropped, which removes affixes such as -able and "
    "recordings of bare punctuation, as are entries containing whitespace, which are read "
    "sentences rather than words. Each word is kept once, since the same word read by two "
    "speakers would otherwise be relevant to only one of its recordings. Files are listed "
    "newest first because alphabetical order puts thousands of digits and symbols before "
    "the first real word. Construction script: scripts/data/lingua_libre/create_data.py."
)

_COMMON = {
    "reference": "https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyMultilingualRetrieval",
    "eval_splits": ["test"],
    "eval_langs": _LANGUAGES,
    "main_score": "ndcg_at_10",
    "date": ("2018-01-01", "2026-09-02"),
    "domains": ["Spoken"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-sa-4.0",
    "annotations_creators": "derived",
    "dialect": [],
    "sample_creation": "found",
    "bibtex_citation": _BIBTEX,
    "is_beta": True,
}


def _load(task: AbsTaskRetrieval, to_text: bool) -> None:
    """Load one direction. `to_text` selects recording->word, otherwise the reverse."""
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    task.dataset = {}
    for lang in _LANGUAGES:
        rows = load_dataset(
            _DATASET_PATH, lang, revision=_DATASET_REVISION, split=split
        )
        audio = rows.select_columns(["id", "audio"])
        text = rows.select_columns(["id", "text"])
        qrels = {i: {i: 1} for i in rows["id"]}

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=audio if to_text else text,
                corpus=text if to_text else audio,
                relevant_docs=qrels,
                top_ranked=None,
            )
        }
    task.data_loaded = True


class LinguaLibreA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LinguaLibreA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the written form of a spoken word. {_CONSTRUCTION}",
        category="a2t",
        modalities=["audio", "text"],
        prompt={"query": "Find the written form of this spoken word."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load(self, to_text=True)


class LinguaLibreT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LinguaLibreT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the recording of a written word. {_CONSTRUCTION}",
        category="t2a",
        modalities=["text", "audio"],
        prompt={"query": "Find the recording of this written word."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load(self, to_text=False)
