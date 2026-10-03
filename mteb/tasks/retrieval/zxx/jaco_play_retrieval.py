from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_JACO_BIBTEX = r"""
@software{dass2023jacoplay,
  author = {Dass, Shivin and Yapeter, Jullian and Zhang, Jesse and Zhang, Jiahui and Pertsch, Karl and Nikolaidis, Stefanos and Lim, Joseph J.},
  title = {{CLVR} Jaco Play Dataset},
  url = {https://github.com/clvrai/clvr_jaco_play_dataset},
  version = {1.0.0},
  year = {2023},
}
"""

_JACO_DESCRIPTION_TAIL = (
    "Built from the CLVR Jaco Play dataset (1,085 teleoperated Jaco arm episodes over 89 "
    "language-annotated tabletop tasks, scene camera at 224x224, 10 fps). Tasks with fewer "
    "than 4 demonstrations are dropped, leaving 65 tasks. Per task, 2 episodes are held out "
    "to supply queries and the remaining episodes form the corpus, so a query's own episode "
    "never appears in the corpus and exact frame matching cannot solve the task. A corpus "
    "item is relevant if and only if it comes from an episode of the same task as the query; "
    "episodes recorded in the same scene that accomplish a different goal are non-relevant, "
    "so matching the scene alone is not enough."
)


class JacoPlayI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JacoPlayI2VRetrieval",
        description=(
            "Image-to-video retrieval over robot manipulation episodes: given a goal-state "
            "image (the final frame of a held-out episode), retrieve manipulation videos "
            "that accomplish the same task. " + _JACO_DESCRIPTION_TAIL
        ),
        reference="https://github.com/clvrai/clvr_jaco_play_dataset",
        dataset={
            "path": "vnahata/JacoPlay-I2V",
            "revision": "50d295dc35edd70b0d2d103a7aecb853b926776f",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_JACO_BIBTEX,
        prompt={
            "query": (
                "Retrieve robot manipulation videos that accomplish the task whose "
                "completed goal state is shown in the image."
            )
        },
    )


class JacoPlayV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JacoPlayV2IRetrieval",
        description=(
            "Video-to-image retrieval over robot manipulation episodes: given a "
            "manipulation video, retrieve goal-state images (final frames of held-out "
            "episodes) of the same task. " + _JACO_DESCRIPTION_TAIL
        ),
        reference="https://github.com/clvrai/clvr_jaco_play_dataset",
        dataset={
            "path": "vnahata/JacoPlay-V2I",
            "revision": "e3937da6f16d5e25391f741621dc8f8fcf46ad02",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_JACO_BIBTEX,
        prompt={
            "query": (
                "Retrieve goal-state images of the task accomplished in the robot "
                "manipulation video."
            )
        },
    )
