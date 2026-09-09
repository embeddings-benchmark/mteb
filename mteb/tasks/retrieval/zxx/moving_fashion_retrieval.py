from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MovingFashionI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MovingFashionI2VRetrieval",
        description=(
            "Shop-image-to-video fashion retrieval obtained by reversing the "
            "official MovingFashion test associations. The benchmark contains "
            "1,340 available e-commerce product-image queries, a global corpus of "
            "1,328 social videos, and 1,341 binary relevance judgments. Repeated "
            "source media paths preserve one genuine multi-positive image query. "
            "The official archive omits one annotated test video, so the shop image "
            "whose only positive video is unavailable is excluded with that qrel."
        ),
        reference="https://arxiv.org/abs/2110.02627",
        dataset={
            "path": "pranitchawla/MovingFashionI2VRetrieval",
            "revision": "29906095b736d5319f14ec1800640b40612b6e8d",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2021-10-06", "2022-01-08"),
        domains=["E-commerce", "Social"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{godi2021movingfashion,
  archiveprefix = {arXiv},
  author = {Godi, Marco and Joppi, Christian and Skenderi, Geri and Cristani, Marco},
  eprint = {2110.02627},
  primaryclass = {cs.CV},
  title = {MovingFashion: a Benchmark for the Video-to-Shop Challenge},
  year = {2021},
}
""",
        prompt={
            "query": (
                "Retrieve social videos showing the clothing item in this shop image."
            )
        },
        is_beta=True,
    )


class MovingFashionV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MovingFashionV2IRetrieval",
        description=(
            "Video-to-shop-image fashion retrieval on the official MovingFashion "
            "test split. The benchmark contains 1,328 available social-video queries, "
            "a global corpus of 1,341 e-commerce product images, and 1,341 binary "
            "relevance judgments. Repeated source media paths preserve 13 genuine "
            "multi-positive video queries. The official archive omits one annotated "
            "test video; that unavailable query and qrel are excluded while its "
            "product image remains in the corpus as a distractor."
        ),
        reference="https://arxiv.org/abs/2110.02627",
        dataset={
            "path": "pranitchawla/MovingFashion",
            "revision": "29c9813e2826ef2f4398455528881ab3e181311b",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["video", "image"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2021-10-06", "2022-01-08"),
        domains=["E-commerce", "Social"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{godi2021movingfashion,
  archiveprefix = {arXiv},
  author = {Godi, Marco and Joppi, Christian and Skenderi, Geri and Cristani, Marco},
  eprint = {2110.02627},
  primaryclass = {cs.CV},
  title = {MovingFashion: a Benchmark for the Video-to-Shop Challenge},
  year = {2021},
}
""",
        prompt={
            "query": (
                "Retrieve the shop image of the clothing item worn in this video."
            )
        },
        is_beta=True,
    )
