from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata
from datasets import Dataset

BIBTEX = r"""
@misc{hong2025worldsense,
  archiveprefix = {arXiv},
  author = {Jack Hong and Shilin Yan and Jiayin Cai and Xiaolong Jiang and Yao Hu and Weidi Xie},
  eprint = {2502.04326},
  primaryclass = {cs.CV},
  title = {WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs},
  url = {https://arxiv.org/abs/2502.04326},
  year = {2025},
}
"""

DATASET = {
    "path": "mteb/WorldSense_1min",
    "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
}

DESCRIPTION_BASE = (
    "Clustering of one-minute real-world clips into eight coarse domains "
    "(e.g. daily life, music, sports), from the WorldSense benchmark "
    "(deduplicated to one sample per video_id)."
)


def _dedupe_one_per_video_id(ds: Dataset) -> Dataset:
    video_ids = ds.select_columns(["video_id"])["video_id"]
    seen: set[str] = set()
    keep_indices: list[int] = []
    for i, vid in enumerate(video_ids):
        if vid not in seen:
            seen.add(vid)
            keep_indices.append(i)
    return ds.select(keep_indices)


class WorldSense1MinDomainAudioVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WorldSense1MinDomainAudioVideoClustering",
        description=DESCRIPTION_BASE + " Uses synchronized video and audio.",
        reference="https://arxiv.org/abs/2502.04326",
        dataset=DATASET,
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2025-01-01", "2025-12-31"),
        domains=["Scene", "Web", "Entertainment"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name: str = "domain"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            ds = _dedupe_one_per_video_id(ds)
            self.dataset[split] = ds.select_columns(
                ["video", "audio", "domain"],
            )


class WorldSense1MinDomainVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WorldSense1MinDomainVideoClustering",
        description=DESCRIPTION_BASE + " Uses video only.",
        reference="https://arxiv.org/abs/2502.04326",
        dataset=DATASET,
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2025-01-01", "2025-12-31"),
        domains=["Scene", "Web", "Entertainment"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "domain"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            ds = _dedupe_one_per_video_id(ds)
            self.dataset[split] = ds.select_columns(
                ["video", "domain"],
            )
