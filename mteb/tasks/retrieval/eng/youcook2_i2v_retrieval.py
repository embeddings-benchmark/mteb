from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{zhou2018towards,
  author = {Zhou, Luowei and Xu, Chenliang and Corso, Jason J.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  title = {Towards Automatic Learning of Procedures from Web Instructional Videos},
  year = {2018},
}
"""

_DESCRIPTION_TAIL = (
    "Built from the YouCook2 dataset of instructional cooking videos "
    "(84 recipes with >= 3 distinct video demonstrations each). "
    "For each recipe, 1 video is held out as a query (its final plated-dish "
    "frame serves as the goal image) and 2-3 other videos from different "
    "cooks form the corpus, so the query's own video never appears in the "
    "corpus and exact frame matching cannot solve the task. A corpus item "
    "is relevant if and only if it demonstrates the same recipe as the query."
)


class YouCook2I2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2I2VRetrieval",
        description=(
            "Image-to-video retrieval over instructional cooking tutorial videos: "
            "given a goal dish image (the final plated dish frame of a held-out "
            "video), retrieve cooking videos demonstrating how to prepare that "
            "dish. " + _DESCRIPTION_TAIL
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={
            "path": "iamfortytwo/YouCook2-I2V",
            "revision": "5a1fa24fd7c3257f760f49db219eeb867fd4ba2f",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Instructional", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": (
                "Retrieve instructional cooking videos that demonstrate how "
                "to prepare the dish shown in the image."
            )
        },
        is_beta=True,
    )


class YouCook2V2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="YouCook2V2IRetrieval",
        description=(
            "Video-to-image retrieval over instructional cooking tutorial videos: "
            "given an instructional cooking video, retrieve goal dish images of "
            "the completed plated dish. " + _DESCRIPTION_TAIL
        ),
        reference="https://arxiv.org/abs/1703.09788",
        dataset={
            "path": "iamfortytwo/YouCook2-V2I",
            "revision": "41b30b00ac44e91215c510da20c2fb36df85ad4d",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["video", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Instructional", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": (
                "Retrieve images of the completed dish prepared in the "
                "cooking tutorial video."
            )
        },
        is_beta=True,
    )
