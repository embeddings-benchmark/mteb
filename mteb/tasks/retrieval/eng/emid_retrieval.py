from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://doi.org/10.1145/3607541.3616821"
_BIBTEX = r"""
@inproceedings{zou2023emid,
  author = {Zou, Jialing and Mei, Jiahao and Ye, Guangze and Huai, Tianyu and Shen, Qiwei and Dong, Daoguo},
  booktitle = {Proceedings of the 1st International Workshop on Multimedia Content Generation and Evaluation: New Methods and Practice},
  doi = {10.1145/3607541.3616821},
  pages = {41--48},
  title = {{EMID}: An Emotional Aligned Dataset in Audio-Visual Modality},
  year = {2023},
}
"""
_DESCRIPTION = (
    "EMID (Emotionally paired Music and Image Dataset) is an audio-visual dataset "
    "for emotional matching of music and images. It contains music clips each "
    "paired with three images aligned under a 13-dimension emotion model, "
    "emphasizing emotional consistency rather than purely semantic correlation."
)


class EMIDA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EMIDA2IRetrieval",
        description=_DESCRIPTION
        + " Queries are music clips and the corpus contains images; each audio clip "
        "is relevant to its three emotion-matched images.",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/EMID-A2I",
            "revision": "ad98feb363b0745966e7722f6af11944c6c90571",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-08-01"),
        domains=["Music", "Web"],
        task_subtypes=["Cross-Modal Retrieval", "Emotion classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve images that match the emotion expressed in this music clip."
        },
        is_beta=True,
    )


class EMIDI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EMIDI2ARetrieval",
        description=_DESCRIPTION
        + " Queries are images and the corpus contains music clips; each image is "
        "relevant to its paired music clip.",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/EMID-I2A",
            "revision": "a917e98ded77d0576302d1ad2e99d9f970618537",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-08-01"),
        domains=["Music", "Web"],
        task_subtypes=["Cross-Modal Retrieval", "Emotion classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the music clip that matches the emotion expressed in this image."
        },
        is_beta=True,
    )
