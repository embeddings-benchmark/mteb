from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ABOI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ABOI2VRetrieval",
        description=(
            "Image-to-video product retrieval from Amazon Berkeley Objects (ABO). "
            "For each product the corpus holds a 360-degree turntable spin and the "
            "query is a separate catalog photograph of the same product, shot at a "
            "different time under different lighting, so a query is never a frame "
            "of its own positive video. Corpus frames are selected by azimuth % 3 "
            "== 0, giving 24 frames at 15-degree steps. Queries are in-situ "
            "photographs; boilerplate, fabric swatches, macro crops and "
            "studio-on-white shots are excluded. Restricted to five volumetric "
            "home-goods types. Data (c) Amazon.com, CC BY 4.0; see the dataset "
            "card for full attribution."
        ),
        reference="https://arxiv.org/abs/2110.06199",
        dataset={
            "path": "hubxrt/ABO-I2V",
            "revision": "579f00a6d09a8b10295bc986911f832882577d62",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2021-06-14", "2021-06-14"),
        domains=["E-commerce", "Scene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{collins2022abo,
  author = {Collins, Jasmine and Goel, Shubham and Deng, Kenan and Luthra, Achleshwar and Xu, Leon and Gundogdu, Erhan and Zhang, Xi and Yago Vicente, Tomas F and Dideriksen, Thomas and Arora, Himanshu and Guillaumin, Matthieu and Malik, Jitendra},
  journal = {CVPR},
  title = {ABO: Dataset and Benchmarks for Real-World 3D Object Understanding},
  year = {2022},
}
""",
        prompt={
            "query": "Retrieve the 360-degree turntable video of the product shown in the photograph."
        },
    )


class ABOV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ABOV2IRetrieval",
        description=(
            "Video-to-image product retrieval from Amazon Berkeley Objects (ABO), "
            "the reverse direction of ABOI2VRetrieval over the same media. Each "
            "query is a product's 360-degree turntable spin (24 frames at "
            "15-degree steps) and its single relevant document is a separate "
            "in-situ catalog photograph of the same product, shot at a different "
            "time under different lighting, so the target is never a frame of "
            "the query video. Restricted to five volumetric home-goods types. "
            "Data (c) Amazon.com, CC BY 4.0; see the dataset card for full "
            "attribution."
        ),
        reference="https://arxiv.org/abs/2110.06199",
        dataset={
            "path": "mteb/ABO-V2I",
            "revision": "d31ebffb86ba0517494fae432f88c53c5cb3d29e",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["video", "image"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2021-06-14", "2021-06-14"),
        domains=["E-commerce", "Scene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{collins2022abo,
  author = {Collins, Jasmine and Goel, Shubham and Deng, Kenan and Luthra, Achleshwar and Xu, Leon and Gundogdu, Erhan and Zhang, Xi and Yago Vicente, Tomas F and Dideriksen, Thomas and Arora, Himanshu and Guillaumin, Matthieu and Malik, Jitendra},
  journal = {CVPR},
  title = {ABO: Dataset and Benchmarks for Real-World 3D Object Understanding},
  year = {2022},
}
""",
        prompt={
            "query": "Retrieve the catalog photograph of the product shown in this 360-degree turntable video."
        },
    )
