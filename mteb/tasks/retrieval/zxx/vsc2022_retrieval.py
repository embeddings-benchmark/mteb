from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VSC2022Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VSC2022Retrieval",
        description=(
            "Video-to-video copy-detection retrieval from the Meta Video Similarity "
            "Challenge (VSC2022). The query is a video that may contain an edited copy "
            "of a segment from a corpus video, and the goal is to retrieve the source "
            "video the query copies from. Matching must be robust to the transformations "
            "used in copy generation rather than semantic similarity. This is a seeded "
            "eval-scale subset of the validation split: 1,926 queries that have a "
            "ground-truth copy source and a corpus of 3,000 reference videos (gold "
            "sources plus seeded distractors), re-encoded to 360p. Videos are "
            "Creative-Commons clips from YFCC100M."
        ),
        reference="https://doi.org/10.1016/j.cviu.2024.103997",
        dataset={
            "path": "dukesun99/VSC2022",
            "revision": "ba6224b845a2181d3de4199aa4eeb8b5b874a1a3",
        },
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="map_at_10",
        date=("2022-01-01", "2023-06-15"),
        domains=["Web"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{pizzi2023vsc,
  author = {Pizzi, Ed and Kordopatis-Zilos, Giorgos and Patel, Hiral and Postelnicu, Gheorghe and Nagavara Ravindra, Sugosh and Gupta, Akshay and Papadopoulos, Symeon and Tolias, Giorgos and Douze, Matthijs},
  doi = {10.1016/j.cviu.2024.103997},
  issn = {1077-3142},
  journal = {Computer Vision and Image Understanding},
  month = {June},
  pages = {103997},
  publisher = {Elsevier BV},
  title = {The 2023 video similarity dataset and challenge},
  url = {https://doi.org/10.1016/j.cviu.2024.103997},
  volume = {243},
  year = {2024},
}
""",
        prompt={"query": "Retrieve the video that this query is a copy of."},
    )
