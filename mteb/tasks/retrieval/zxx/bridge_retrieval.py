from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BRIDGE_BIBTEX = r"""
@inproceedings{walke2023bridgedata,
  author = {Homer Walke and Kevin Black and Abraham Lee and Moo Jin Kim and Max Du and Chongyi Zheng and Tony Zhao and Philippe Hansen-Estruch and Quan Vuong and Andre He and Vivek Myers and Kuan Fang and Chelsea Finn and Sergey Levine},
  booktitle = {Conference on Robot Learning (CoRL)},
  title = {BridgeData V2: A Dataset for Robot Learning at Scale},
  year = {2023},
}
"""


class BridgeV2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BridgeV2VRetrieval",
        description=(
            "Cross-viewpoint video-to-video retrieval over real-world robot "
            "manipulation: given a manipulation episode video from an "
            "alternate camera, retrieve the video of the same episode "
            "captured by the main over-shoulder camera. Built from "
            "BridgeData V2 (WidowX tabletop manipulation across many "
            "scenes): episodes carrying a real second viewpoint are "
            "filtered to 3-60 s, deduplicated by language instruction (one "
            "episode per unique instruction, which also spreads scenes), "
            "and evenly subsampled to 1,500. Queries and documents never "
            "share a viewpoint, so exact frame matching cannot solve the "
            "task; relevance is instance-level 1:1."
        ),
        reference="https://rail-berkeley.github.io/bridgedata/",
        dataset={
            "path": "ZhixuLi/BridgeData-V2V",
            "revision": "ee5a70b3f9242b328f134606b524dca7cb3c1fd0",
        },
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2023-08-31"),
        domains=["Robotics", "Scene"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BRIDGE_BIBTEX,
        prompt={
            "query": (
                "Retrieve the robot manipulation video showing the same "
                "episode from a different viewpoint."
            )
        },
    )
