from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_METAWORLD_BIBTEX = r"""
@inproceedings{yu2019meta,
  author = {Yu, Tianhe and Quillen, Deirdre and He, Zhanpeng and Julian, Ryan and Hausman, Karol and Finn, Chelsea and Levine, Sergey},
  booktitle = {Conference on Robot Learning (CoRL)},
  title = {{Meta-World}: A Benchmark and Evaluation for Multi-Task and Meta-Reinforcement Learning},
  year = {2019},
}
"""

_METAWORLD_DESCRIPTION_TAIL = (
    "Built from the Meta-World benchmark for multi-task robot learning "
    "(MT50 benchmark: 49 tabletop manipulation tasks with simulated Sawyer robot, "
    "teleoperated demonstrations rendered at 256x256, 10 fps). "
    "Per task, 5 episodes are held out as queries and 10 episodes form the corpus, "
    "so the query's own episode never appears in the corpus and exact frame matching "
    "cannot solve the task. A corpus item is relevant if and only if it comes from an "
    "episode of the same task as the query (10 relevant items per query); every other "
    "item is non-relevant."
)


class MetaWorldMT50I2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MetaWorldMT50I2VRetrieval",
        description=(
            "Image-to-video retrieval over robot manipulation episodes: given "
            "a goal-state image (the final frame of a held-out episode), "
            "retrieve manipulation videos that accomplish the same task. "
            + _METAWORLD_DESCRIPTION_TAIL
        ),
        reference="https://meta-world.github.io/",
        dataset={
            "path": "iamfortytwo/MetaWorld-MT50-I2V",
            "revision": "3feedd96246340c75c5191db7cabbe0d9fb3c27d",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2019-10-01", "2020-05-01"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="rendered",
        is_beta=True,
        bibtex_citation=_METAWORLD_BIBTEX,
        prompt={
            "query": (
                "Retrieve robot manipulation videos that accomplish the task "
                "whose completed goal state is shown in the image."
            )
        },
    )


class MetaWorldMT50V2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MetaWorldMT50V2IRetrieval",
        description=(
            "Video-to-image retrieval over robot manipulation episodes: given "
            "a manipulation video, retrieve goal-state images (final frames "
            "of held-out episodes) of the same task. "
            + _METAWORLD_DESCRIPTION_TAIL
        ),
        reference="https://meta-world.github.io/",
        dataset={
            "path": "iamfortytwo/MetaWorld-MT50-V2I",
            "revision": "ee03d6c2f978924bb31325cc5241fb0a589f0287",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["video", "image"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2019-10-01", "2020-05-01"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="rendered",
        is_beta=True,
        bibtex_citation=_METAWORLD_BIBTEX,
        prompt={
            "query": (
                "Retrieve goal-state images showing the completed outcome of "
                "the task performed in the robot manipulation video."
            )
        },
    )
