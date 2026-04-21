from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from ._video_pair_helpers import build_pair_dataset, generate_pairs


class AVSpeakerBenchPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AVSpeakerBenchPairClassification",
        description=(
            "Pair classification on AV-SpeakerBench: determining whether "
            "two video clips come from the same source video (same speaker "
            "context) or different source videos (different speakers). "
            "Clips are grouped by their YouTube source video ID, so pairs "
            "from the same source share speakers and visual context."
        ),
        reference="https://arxiv.org/abs/2512.02231",
        dataset={
            "path": "mteb/AV-SpeakerBench",
            "revision": "46da53344c6a968d30cd72e28862732adf747802",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2025-12-01", "2025-12-31"),
        domains=["Spoken"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{nguyen2025avspeakerbench,
  author = {Nguyen, Le Thien Phuc and Yu, Zhuoran and Hang, Samuel Low Yu and An, Subin and Lee, Jeongik and Ban, Yohan and Chung, SeungEun and Nguyen, Thanh-Huy and others},
  journal = {arXiv preprint arXiv:2512.02231},
  title = {See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models},
  year = {2025},
}
""",
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        import random

        rng = random.Random(42)
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]

            # Deduplicate rows by video_id to avoid inflated positive pairs
            video_ids = ds["video_id"]
            seen: set[str] = set()
            unique_indices: list[int] = []
            for i, vid in enumerate(video_ids):
                if vid not in seen:
                    seen.add(vid)
                    unique_indices.append(i)
            ds = ds.select(unique_indices)

            # Extract source video IDs from video_id field
            # Format: h_O8ZvB3uEk_47_59.mp4 → h_O8ZvB3uEk (YouTube ID)
            video_ids = ds["video_id"]
            source_ids: list[str] = []
            for vid in video_ids:
                # Remove .mp4, split by _, remove last two parts (start, end)
                parts = vid.replace(".mp4", "").split("_")
                source_id = "_".join(parts[:-2])
                source_ids.append(source_id)

            # Map source IDs to integer labels for generate_pairs
            unique_sources = sorted(set(source_ids))
            source_to_label = {s: i for i, s in enumerate(unique_sources)}
            class_labels = [source_to_label[s] for s in source_ids]

            pairs = generate_pairs(class_labels, rng)
            self.dataset[split] = build_pair_dataset(ds, pairs)
