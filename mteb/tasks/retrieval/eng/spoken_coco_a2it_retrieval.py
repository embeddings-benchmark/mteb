from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE_SPOKENCOCO = "https://aclanthology.org/2021.acl-long.411/"
_BIBTEX = r"""
@inproceedings{hsu-etal-2021-text,
  author = {Hsu, Wei-Ning and Harwath, David and Miller, Tyler and Song, Christopher and Glass, James},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.411},
  pages = {5284--5300},
  title = {Text-Free Image-to-Speech Synthesis Using Learned Segmental Units},
  year = {2021},
}
@misc{lin2014microsoftcococommonobjects,
  title = {Microsoft COCO: Common Objects in Context},
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C. Lawrence},
  year = {2014},
  eprint = {1405.0312},
  archivePrefix = {arXiv},
}
"""
_DESCRIPTION = (
    "SpokenCOCO pairs MS COCO images with recordings of human speakers reading "
    "the corresponding English captions. This task uses the 5,000-image Karpathy "
    "test split with 25,031 spoken captions. "
    "Queries are spoken audio captions; corpus items contain both the MS COCO image "
    "and its written text caption, requiring models to process all three modalities "
    "(audio, image, text) to fully exploit the corpus signal."
)


class SpokenCOCOA2ITRetrieval(AbsTaskRetrieval):
    metadata: TaskMetadata = TaskMetadata(
        name="SpokenCOCOA2ITRetrieval",
        description=_DESCRIPTION,
        reference=_REFERENCE_SPOKENCOCO,
        dataset={
            "path": "rakshi719/SpokenCOCO-A2IT",
            "revision": "f978bafbcb6720bac95803346c2c0cf40fec5a25",
        },
        type="Any2AnyRetrieval",
        category="a2it",
        modalities=["audio", "image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2014-01-01", "2020-12-03"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the (image, text caption) pair described by the spoken audio."},
        is_beta=True,
    )
