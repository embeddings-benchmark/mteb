from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://aclanthology.org/2021.acl-long.411/"
_BIBTEX = r"""
@inproceedings{hsu-etal-2021-text,
  author = {Hsu, Wei-Ning and Harwath, David and Miller, Tyler and Song, Christopher and Glass, James},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.411},
  pages = {5284--5300},
  title = {Text-Free Image-to-Speech Synthesis Using Learned Segmental Units},
  year = {2021},
}
"""
_DESCRIPTION = (
    "SpokenCOCO pairs MS COCO images with recordings of human speakers reading "
    "the corresponding English captions. This task uses the 5,000-image Karpathy "
    "test split with 25,031 spoken captions. "
)


class SpokenCOCOA2IRetrieval(AbsTaskRetrieval):
    metadata: TaskMetadata = TaskMetadata(
        name="SpokenCOCOA2IRetrieval",
        description=_DESCRIPTION
        + "Queries are spoken captions and the corpus contains images; the goal is "
        "to retrieve the image described by each recording.",
        reference=_REFERENCE,
        dataset={
            "path": "whybe-choi/SpokenCOCOA2IRetrieval",
            "revision": "72731c49562b73ea7e1cbd296c40f3924a4e5fad",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2014-01-01", "2020-12-03"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the image described by the spoken caption."},
    )


class SpokenCOCOI2ARetrieval(AbsTaskRetrieval):
    metadata: TaskMetadata = TaskMetadata(
        name="SpokenCOCOI2ARetrieval",
        description=_DESCRIPTION
        + "Queries are images and the corpus contains spoken captions; the goal is "
        "to retrieve the recordings that describe each image.",
        reference=_REFERENCE,
        dataset={
            "path": "whybe-choi/SpokenCOCOI2ARetrieval",
            "revision": "d9bb53ec91142fbc8a1bcefea200d835bd916a8b",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2014-01-01", "2020-12-03"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the spoken captions that describe this image."},
    )
