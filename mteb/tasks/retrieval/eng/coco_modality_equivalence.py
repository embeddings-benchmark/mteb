"""COCO Modality Equivalence retrieval tasks.

Six retrieval directions over a shared 1k-image MSCOCO pool where every item
is simultaneously available as an image, a text caption, a human-spoken
caption (SpokenCOCO) and a TTS-synthesized caption (SpeechCoco).

Because the candidate pool is IDENTICAL across all directions, any difference
in retrieval score is attributable to modality difficulty rather than dataset
content. This directly answers the question posed by Adnan in issue #5358:
"we can't currently separate 'this modality pair is hard' from 'this dataset
is hard'."

Directions:
  t2i  -- text caption  -> image          (compare with a2i_h / a2i_s)
  a2i_h -- human speech -> image
  a2i_s -- TTS speech   -> image
  i2t  -- image         -> text caption
  i2a_h -- image        -> human speech
  i2a_s -- image        -> TTS speech
"""

from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "rakshi719/coco-modality-equivalence"
_DATASET_REVISION = "3aabaa047c92980e12628149df52425757b3f8a6"

_REFERENCE = "https://github.com/embeddings-benchmark/mteb/issues/5358"

_BIBTEX = r"""
@inproceedings{havard2017speechcoco,
  author = {Havard, William and Besacier, Laurent and Rosec, Olivier},
  booktitle = {GLU 2017 Workshop},
  title = {SPEECH-COCO: 600k Visually Grounded Spoken Captions Aligned to MSCOCO},
  year = {2017},
}

@inproceedings{hsu2021spokencoco,
  author = {Hsu, Wei-Ning and Harwath, David and Miller, Tyler and Song, Christopher and Glass, James},
  booktitle = {Proceedings of ACL-IJCNLP 2021},
  title = {Text-Free Image-to-Speech Synthesis Using Learned Segmental Units},
  year = {2021},
}

@inproceedings{lin2014microsoft,
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle = {Computer Vision--ECCV 2014},
  title = {Microsoft coco: Common objects in context},
  year = {2014},
}
"""

_SHARED_POOL_NOTE = (
    "All six tasks in this group share the same candidate pool of MSCOCO images "
    "paired with text, human-spoken (SpokenCOCO), and TTS (SpeechCoco) captions. "
    "Comparing scores across directions isolates the effect of modality from the "
    "effect of content. "
)


def _dataset(config: str) -> dict:
    return {"path": _DATASET_PATH, "revision": _DATASET_REVISION, "name": config}


_COMMON = dict(
    reference=_REFERENCE,
    type="Any2AnyRetrieval",
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=("2014-01-01", "2021-12-31"),
    domains=["Scene", "Spoken"],
    task_subtypes=["Cross-Modal Retrieval"],
    license="cc-by-sa-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=_BIBTEX,
    is_beta=True,
)


class COCOModalEquivT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="COCOModalEquivT2IRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are text captions; the corpus contains images. "
        "Baseline direction for comparing against spoken-audio queries "
        "(COCOModalEquivA2IHumanRetrieval, COCOModalEquivA2ITTSRetrieval).",
        category="t2i",
        modalities=["text", "image"],
        prompt={"query": "Find the image described by the caption."},
        dataset=_dataset("t2i"),
        **_COMMON,
    )


class COCOModalEquivA2IHumanRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="COCOModalEquivA2IHumanRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are human-spoken captions (SpokenCOCO recordings); the corpus "
        "contains images. Compare with COCOModalEquivT2IRetrieval and "
        "COCOModalEquivA2ITTSRetrieval to measure the cost of the speech modality.",
        category="a2i",
        modalities=["audio", "image"],
        prompt={"query": "Find the image described by the spoken caption."},
        dataset=_dataset("a2i_h"),
        **_COMMON,
    )


class COCOModalEquivA2ITTSRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="COCOModalEquivA2ITTSRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are TTS-synthesized captions (SpeechCoco); the corpus contains "
        "images. Compare with COCOModalEquivA2IHumanRetrieval to measure the gap "
        "between natural and synthetic speech on the same content.",
        category="a2i",
        modalities=["audio", "image"],
        prompt={"query": "Find the image described by the spoken caption."},
        dataset=_dataset("a2i_s"),
        **_COMMON,
    )


class COCOModalEquivI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="COCOModalEquivI2TRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are images; the corpus contains text captions. "
        "Baseline direction for comparing against image-to-audio retrieval.",
        category="i2t",
        modalities=["image", "text"],
        prompt={"query": "Find the caption that describes this image."},
        dataset=_dataset("i2t"),
        **_COMMON,
    )


class COCOModalEquivI2AHumanRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="COCOModalEquivI2AHumanRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are images; the corpus contains human-spoken captions (SpokenCOCO). "
        "Compare with COCOModalEquivI2TRetrieval to measure the cost of retrieving "
        "into an audio corpus vs a text corpus.",
        category="i2a",
        modalities=["image", "audio"],
        prompt={"query": "Find the spoken caption that describes this image."},
        dataset=_dataset("i2a_h"),
        **_COMMON,
    )


class COCOModalEquivI2ATTSRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="COCOModalEquivI2ATTSRetrieval",
        description=_SHARED_POOL_NOTE
        + "Queries are images; the corpus contains TTS captions (SpeechCoco). "
        "Compare with COCOModalEquivI2AHumanRetrieval to measure human-vs-TTS "
        "audio corpus difficulty for the same image queries.",
        category="i2a",
        modalities=["image", "audio"],
        prompt={"query": "Find the spoken caption that describes this image."},
        dataset=_dataset("i2a_s"),
        **_COMMON,
    )
