from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from datasets import concatenate_datasets, load_dataset

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping


class VELOCITIPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VELOCITIPairClassification",
        description=(
            "Tests whether a model correctly binds actions to the agents performing "
            "them across a video, by pairing each video with a correct caption and a "
            "subtly wrong one and measuring how well embedding similarity separates "
            "the two. Negatives are constructed via in-video negation (swapping "
            "agents/actions that both occur in the same video) or text-inspired "
            "negation (LLM-generated contradictions), forming a strict entailment "
            "test rather than a coarse mismatch. From the VELOCITI benchmark."
        ),
        reference="https://arxiv.org/abs/2406.10889",
        dataset={
            "path": "yaswanth169/VELOCITI-PC",
            "revision": "dfbe014eacf4507cb0fad6a66497f43d30ce8044",
        },
        type="VideoPairClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-06-16", "2024-06-16"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="LM-generated and verified",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{saravanan2025velociti,
  author = {Saravanan, Darshana and Gupta, Varun and Singh, Darshan and Khan, Zeeshan and Gandhi, Vineet and Tapaswi, Makarand},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title = {VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment},
  year = {2025},
}
""",
    )

    input1_column_name: ClassVar[Mapping[str, str]] = {"video": "video"}
    input2_column_name: ClassVar[Mapping[str, str]] = {"text": "text"}
    label_column_name: str = "label"

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        # (video_id, text, label), 11,669 rows -- one per pos/neg caption.
        raw = load_dataset(path, revision=revision, split="test")
        # (video_id, video), 864 unique rows -- no per-row video duplication.
        videos_ds = load_dataset(path, "videos", revision=revision, split="test")

        video_id_to_idx = {vid: i for i, vid in enumerate(videos_ds["video_id"])}
        aligned_indices = [video_id_to_idx[vid] for vid in raw["video_id"]]
        aligned_videos = videos_ds.select(aligned_indices).select_columns(["video"])

        self.dataset = {"test": concatenate_datasets([raw, aligned_videos], axis=1)}
        self.data_loaded = True
