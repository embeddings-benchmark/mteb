from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ABOI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ABOI2VRetrieval",
        description=(
            "Image-to-video product retrieval built from Amazon Berkeley Objects "
            "(ABO). ABO ships, for the same product, two independently captured "
            "assets: a 360-degree turntable 'spin' photographed in a rig, and "
            "separate catalog photographs shot at a different time, in a different "
            "place, under different lighting. A query is therefore never a frame of "
            "its own positive video, and no crop, re-encode or temporal-neighbour "
            "relationship links the two, so frame leakage is structurally impossible "
            "rather than filtered out after the fact. "
            "The corpus holds one video per product, encoded from that product's "
            "spin by selecting azimuth % 3 == 0, which yields exactly 24 frames at "
            "15-degree steps for every spin in ABO and so keeps the corpus "
            "homogeneous without dropping any sequence. Each query is one catalog "
            "photograph of the same product taken in a room or scene, so retrieval "
            "must bridge a styled in-situ photograph and a turntable render on a "
            "white sweep. Catalog images whose perceptual hash is shared with "
            "another product (boilerplate, size charts), that fail a zero-shot "
            "category gate (fabric swatches, macro crops, dimension diagrams), that "
            "fall within a perceptual-hash radius of any frame of the product's own "
            "spin, or that are studio shots on a white background are all excluded "
            "from the query set. Products are restricted to five volumetric "
            "home-goods types (CHAIR, SOFA, TABLE, HOME_FURNITURE_AND_DECOR, LAMP); "
            "flat goods such as rugs and wall art are excluded because a turntable "
            "rotation of a flat object is close to degenerate. One product per spin "
            "sequence, so no two queries share a relevant document. "
            "Credit for the data, including all images, must be given to Amazon.com. "
            "Credit for building the dataset, archives and benchmark sets must be "
            "given to Matthieu Guillaumin (Amazon.com), Thomas Dideriksen "
            "(Amazon.com), Kenan Deng (Amazon.com), Himanshu Arora (Amazon.com), "
            "Arnab Dhua (Amazon.com), Xi (Brian) Zhang (Amazon.com), Tomas "
            "Yago-Vicente (Amazon.com), Jasmine Collins (UC Berkeley), Shubham Goel "
            "(UC Berkeley) and Jitendra Malik (UC Berkeley). The turntable frame "
            "sequences were re-encoded to video and the catalog images were filtered "
            "and subset-selected; no source pixels were otherwise altered."
        ),
        reference="https://arxiv.org/abs/2110.06199",
        dataset={
            "path": "hubxrt/ABO-I2V",
            "revision": "d99aa44bf13b322748fc16faf0ec7489ed3653db",
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
