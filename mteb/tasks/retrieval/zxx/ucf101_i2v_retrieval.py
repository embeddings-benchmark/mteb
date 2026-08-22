from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class UCF101I2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UCF101I2VRetrieval",
        description=(
            "Image-to-video action retrieval built from the UCF101 official split-1 "
            "test set. UCF101 clips are grouped by scene (v_<Class>_g<NN>_c<NN>, "
            "where gNN is one actor/scene/recording session), and clips inside a "
            "group are near-duplicates. The corpus therefore holds exactly one clip "
            "per group (c01), so it contains no intra-group near-duplicates. Each "
            "query is a single frame sampled from the interior (25-75% of the "
            "duration) of the highest-index clip of the same group, always at least "
            "two clip indices away from the corpus clip. Groups whose query frame "
            "remains near-identical to a frame of its own positive clip are dropped, "
            "leaving 337 query/clip pairs over all 51 classes."
        ),
        reference="https://arxiv.org/abs/1212.0402",
        dataset={
            "path": "hubxrt/UCF101-I2V",
            "revision": "f252bcab74dd18e01f7e3f2d62050400d261fe1d",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2012-01-01", "2012-12-03"),
        domains=["Activity", "Sport", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@misc{Soomro2012UCF101,
  archiveprefix = {arXiv},
  author = {Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  eprint = {1212.0402},
  primaryclass = {cs.CV},
  title = {UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  url = {https://arxiv.org/abs/1212.0402},
  year = {2012},
}
""",
        prompt={
            "query": "Retrieve the video clip containing the scene shown in the image."
        },
    )
