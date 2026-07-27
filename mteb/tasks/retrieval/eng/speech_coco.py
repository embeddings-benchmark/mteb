from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/1707.08435"
_BIBTEX = r"""
@inproceedings{havard2017speechcoco,
  author = {Havard, William and Besacier, Laurent and Rosec, Olivier},
  booktitle = {Proceedings of the GLU 2017 International Workshop on Grounding Language Understanding},
  title = {SPEECH-COCO: 600k Visually Grounded Spoken Captions Aligned to MSCOCO Data Set},
  year = {2017},
}
"""
_DESCRIPTION = (
    "SPEECH-COCO pairs MS-COCO images with spoken captions synthesized using eight "
    "high-quality commercial TTS voices with varied speaking rates and injected "
    "disfluencies. This retrieval task uses a deterministic downsample of the "
    "validation split: 2,048 images sampled from the first eight validation shards "
    "with 1,000 queries. "
)


class SpeechCocoA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpeechCocoA2IRetrieval",
        description=_DESCRIPTION
        + "Queries are spoken captions and the corpus contains the images; the goal "
        "is to retrieve the image described by the spoken caption.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/SpeechCoco-A2I",
            "revision": "217c6660258de6e60002f748abdf11be623c8e0e",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2014-01-01", "2017-07-31"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the image described by the spoken caption."},
    )


class SpeechCocoI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpeechCocoI2ARetrieval",
        description=_DESCRIPTION
        + "Queries are images and the corpus contains spoken captions; the goal is "
        "to retrieve the spoken caption describing the image.",
        reference=_REFERENCE,
        dataset={
            "path": "dukesun99/SpeechCoco-I2A",
            "revision": "afb9e08254be4e9c2e5af432912291dde7528b68",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2014-01-01", "2017-07-31"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the spoken caption that describes the image."},
    )
