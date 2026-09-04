from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/SpokenWikipedia-retrieval"
_DATASET_REVISION = "edaf5feb7f04a64a1a49e54660bc5b5dcc35f7f7"

_LANGUAGES = {
    "nld": ["nld-Latn"],
    "eng": ["eng-Latn"],
    "deu": ["deu-Latn"],
    "spa": ["spa-Latn"],
    "fra": ["fra-Latn"],
}

_BIBTEX = r"""
@misc{spokenwikipedia,
  author = {{Wikimedia Commons contributors}},
  howpublished = {\url{https://commons.wikimedia.org/wiki/Category:Spoken_Wikipedia}},
  title = {Spoken {Wikipedia}},
  year = {2026},
}
"""

_DESCRIPTION = (
    "Volunteer readings of Wikipedia articles paired with the article lead, in Dutch, "
    "English, German, Spanish and French. The pairing is which article a recording is of, "
    "so it holds however much of the reading is kept."
)

_CONSTRUCTION = (
    "Recordings come from the Spoken Wikipedia categories on Wikimedia Commons, which is "
    "free by site policy, and the lead text from each Wikipedia, which is CC-BY-SA. "
    "Readings run to tens of minutes, so only the opening 60 seconds is kept: readers "
    "start at the lead, so the opening and the lead describe the same subject. The article "
    "for a recording is found from the file's global usage, restricted to main-namespace "
    "pages on the matching Wikipedia so that project and portal pages are excluded, "
    "falling back to the filename where a file is no longer embedded in its article. Leads "
    "under 200 characters are dropped as too thin to identify, and one recording is kept "
    "per article. Construction script: scripts/data/spoken_wikipedia/create_data.py."
)

_COMMON = {
    "reference": "https://commons.wikimedia.org/wiki/Category:Spoken_Wikipedia",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyMultilingualRetrieval",
    "eval_splits": ["test"],
    "eval_langs": _LANGUAGES,
    "main_score": "ndcg_at_10",
    "date": ("2005-01-01", "2026-09-01"),
    "domains": ["Spoken", "Encyclopaedic"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-sa-4.0",
    "annotations_creators": "derived",
    "dialect": [],
    "sample_creation": "found",
    "bibtex_citation": _BIBTEX,
    "is_beta": True,
}


def _load(task: AbsTaskRetrieval, to_text: bool) -> None:
    """Load one direction. `to_text` selects reading->lead, otherwise the reverse."""
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


class SpokenWikipediaA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpokenWikipediaA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the article a reading is of. {_CONSTRUCTION}",
        category="a2t",
        modalities=["audio", "text"],
        prompt={"query": "Find the article this recording is reading."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load(self, to_text=True)


class SpokenWikipediaT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpokenWikipediaT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the reading of an article. {_CONSTRUCTION}",
        category="t2a",
        modalities=["text", "audio"],
        prompt={"query": "Find the recording that reads this article."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load(self, to_text=False)
