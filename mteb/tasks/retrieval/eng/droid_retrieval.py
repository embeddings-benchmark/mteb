from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DROIDIT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DROIDIT2VRetrieval",
        description=(
            "Composed image+text to video retrieval over real-world robot "
            "manipulation: given the initial scene image and the language "
            "instruction of an episode, retrieve the video of the robot "
            "performing that instruction. Built from the DROID dataset "
            "(large-scale in-the-wild Franka manipulation; 1,500 successful "
            "episodes with unique instructions, 5-60 s at 15 fps). The query "
            "image comes from the exterior_1 camera while corpus videos come "
            "from the exterior_2 camera of the same episodes, so queries and "
            "documents never share a viewpoint and exact frame matching "
            "cannot solve the task; models must ground the instruction and "
            "scene across viewpoints. Relevance is instance-level 1:1."
        ),
        reference="https://droid-dataset.github.io/",
        dataset={
            "path": "ZhixuLi/DROID-IT2V",
            "revision": "ed5e541975b322cc9d9784fafc18d69ab04ef534",
        },
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-03-01", "2024-01-31"),
        domains=["Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{khazatsky2024droid,
  author = {Alexander Khazatsky and Karl Pertsch and Suraj Nair and Ashwin Balakrishna and Sudeep Dasari and Siddharth Karamcheti and Soroush Nasiriany and Mohan Kumar Srirama and Lawrence Yunliang Chen and Kirsty Ellis and others},
  booktitle = {Robotics: Science and Systems},
  title = {{DROID}: A Large-Scale In-the-Wild Robot Manipulation Dataset},
  year = {2024},
}
""",
        prompt={
            "query": (
                "Given the initial scene image and the task instruction, "
                "retrieve the video of the robot performing the instruction "
                "in that scene."
            )
        },
    )
