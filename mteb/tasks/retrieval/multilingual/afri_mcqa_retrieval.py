from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/AfriMCQA-speech-image-retrieval"
_DATASET_REVISION = "e189ac0e5b426bf423c9859a9e7c4d9219cd845b"

_LANGUAGES = {
    "twi": ["twi-Latn"],
    "amh": ["amh-Ethi"],
    "nya": ["nya-Latn"],
    "hau": ["hau-Latn"],
    "ibo": ["ibo-Latn"],
    "kik": ["kik-Latn"],
    "kin": ["kin-Latn"],
    "lin": ["lin-Latn"],
    "lug": ["lug-Latn"],
    # Afri-MCQA's Oromo is the West Central variety, so `gaz` rather than the `orm` macrolanguage
    "orm": ["gaz-Latn"],
    "sot": ["sot-Latn"],
    "tsn": ["tsn-Latn"],
    "som": ["som-Latn"],
    "tir": ["tir-Ethi"],
    "yor": ["yor-Latn"],
    "zul": ["zul-Latn"],
}

_BIBTEX = r"""
@inproceedings{tonja2026afrimcqa,
  author = {Tonja, Atnafu Lambebo and Anand, Srija and Villa-Cueva, Emilio and Azime, Israel Abebe and Alabi, Jesujoba Oluwadara and Mohamed, Muhidin A. and Yadeta, Debela Desalegn and Abadi, Negasi Haile and Oppong, Abigail and Obiefuna, Nnaemeka Casmir and Abdulmumin, Idris and Etori, Naome A},
  title = {{Afri-MCQA}: Multimodal Cultural Question Answering for {African} Languages},
  year = {2026},
}
"""

_DESCRIPTION = (
    "Speech-image retrieval over Afri-MCQA, a culturally grounded benchmark written and "
    "recorded by native speakers of 16 African languages. Questions are spoken aloud in "
    "the native language and grounded in a photograph."
)

_CONSTRUCTION = (
    "Built from the official test split, keeping the question audio and dropping the four "
    "spoken answer options, which describe candidate answers rather than the image. Images "
    "are held per language because one photograph can carry questions in several "
    "languages, so a pooled corpus would score a correct retrieval as wrong. Construction "
    "script: scripts/data/afri_mcqa_retrieval/create_data.py."
)

_COMMON = {
    "reference": "https://arxiv.org/abs/2601.05699",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyMultilingualRetrieval",
    "eval_splits": ["test"],
    "eval_langs": _LANGUAGES,
    "main_score": "ndcg_at_10",
    "date": ("2025-01-01", "2026-01-15"),
    "domains": ["Scene", "Spoken"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-nc-4.0",
    "annotations_creators": "human-annotated",
    "dialect": [],
    "sample_creation": "created",
    "bibtex_citation": _BIBTEX,
    "is_beta": True,
}


def _load_afri_mcqa(task: AbsTaskRetrieval, to_image: bool) -> None:
    """Load one Afri-MCQA direction for every language.

    `to_image` selects question audio -> image; otherwise image -> question audio.
    """
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    task.dataset = {}
    for lang in _LANGUAGES:
        images = load_dataset(
            _DATASET_PATH, f"{lang}-images", revision=_DATASET_REVISION, split=split
        )
        audio = load_dataset(
            _DATASET_PATH, f"{lang}-audio", revision=_DATASET_REVISION, split=split
        )

        # Read the link columns directly; iterating full rows would decode every clip.
        links = audio.select_columns(["id", "image_id"]).to_dict()
        pairs = list(zip(links["id"], links["image_id"], strict=True))

        if to_image:
            queries = audio.select_columns(["id", "audio"])
            corpus = images
            qrels = {qid: {img: 1} for qid, img in pairs}
        else:
            qrels = {}
            for qid, img in pairs:
                qrels.setdefault(img, {})[qid] = 1
            # select() by index rather than filter(), which would decode every image
            wanted = [i for i, id_ in enumerate(images["id"]) if id_ in qrels]
            queries = images.select(wanted)
            corpus = audio.select_columns(["id", "audio"])

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
            )
        }
    task.data_loaded = True


class AfriMCQAA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AfriMCQAA2IRetrieval",
        description=f"{_DESCRIPTION} Retrieve the photograph a spoken question asks about. {_CONSTRUCTION}",
        category="a2i",
        modalities=["audio", "image"],
        prompt={"query": "Find the image that this spoken question is asking about."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_afri_mcqa(self, to_image=True)


class AfriMCQAI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AfriMCQAI2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the spoken question asked about a photograph. {_CONSTRUCTION}",
        category="i2a",
        modalities=["image", "audio"],
        prompt={"query": "Find the spoken question that asks about this image."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_afri_mcqa(self, to_image=False)
