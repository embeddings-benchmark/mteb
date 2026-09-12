from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_GOT10K_BIBTEX = r"""
@article{huang2019got,
  author = {Lianghua Huang and Xin Zhao and Kaiqi Huang},
  title = {{GOT-10k}: A Large High-Diversity Benchmark for Generic Object
           Tracking in the Wild},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2019},
}
"""

_GOT10K_DESCRIPTION_TAIL = (
    "Built from the GOT-10k (Generic Object Tracking) validation split — "
    "180 real-world tracking sequences spanning 563 object classes and "
    "87 motion patterns, covering humans, animals, vehicles, and everyday "
    "objects in indoor/outdoor scenes. Videos are encoded from the official "
    "JPEG frames at 10 fps. The mapping is one-to-one: each sequence "
    "contributes one query and one corpus item, with no distractors, so "
    "models must match fine-grained instance appearance across time."
)


class GOT10kI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GOT10kI2VRetrieval",
        description=(
            "Image-to-video retrieval over generic object tracking sequences: "
            "given the first frame of a tracking sequence (image), retrieve "
            "the full tracking video that follows. " + _GOT10K_DESCRIPTION_TAIL
        ),
        reference="https://arxiv.org/abs/1808.00803",
        dataset={
            "path": "rakshi719/GOT10k-I2V",
            "revision": "b650a55bc4b7fbc25582421eeb4493177397a917",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2019-12-31"),
        domains=["Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_GOT10K_BIBTEX,
        prompt={
            "query": (
                "Retrieve the tracking video that shows the same object instance "
                "as the one in this first-frame image."
            )
        },
    )


class GOT10kV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GOT10kV2IRetrieval",
        description=(
            "Video-to-image retrieval over generic object tracking sequences: "
            "given a tracking video, retrieve its corresponding first frame "
            "(image). " + _GOT10K_DESCRIPTION_TAIL
        ),
        reference="https://arxiv.org/abs/1808.00803",
        dataset={
            "path": "rakshi719/GOT10k-V2I",
            "revision": "027a6f397f5ebb138504906c77c6ddbad27322c2",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2019-12-31"),
        domains=["Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_GOT10K_BIBTEX,
        prompt={
            "query": (
                "Retrieve the first-frame image of the tracking sequence "
                "shown in this video."
            )
        },
    )
