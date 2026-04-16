from __future__ import annotations

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class WorldSense1MinDomainClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WorldSense1MinDomainClustering",
        description=(
            "Clustering of one-minute real-world clips into eight coarse domains "
            "(e.g. daily life, music, sports) using synchronized video and audio, "
            "from the WorldSense benchmark (deduplicated to one sample per video_id)."
        ),
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSense_1min",
            "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
        },
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2025-01-01", "2025-12-31"),
        domains=["Scene", "Web", "Entertainment"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{hong2025worldsense,
  title = {WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs},
  author = {Jack Hong and Shilin Yan and Jiayin Cai and Xiaolong Jiang and Yao Hu and Weidi Xie},
  year = {2025},
  eprint = {2502.04326},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2502.04326},
}
""",
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name: str = "domain"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            # Read only video_id first so parquet column pruning avoids decoding video/audio
            # for the full 1k+ QA rows while deduplicating.
            video_ids = ds.select_columns(["video_id"])["video_id"]
            seen: set[str] = set()
            keep_indices: list[int] = []
            for i, vid in enumerate(video_ids):
                if vid not in seen:
                    seen.add(vid)
                    keep_indices.append(i)
            ds = ds.select(keep_indices)
            self.dataset[split] = ds.select_columns(
                ["video", "audio", "domain"],
            )
